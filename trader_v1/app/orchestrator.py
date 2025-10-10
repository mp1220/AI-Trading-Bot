"""Main orchestrator coordinating data, models, trading, and monitoring."""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from contextlib import suppress
from pathlib import Path
from typing import Optional

from app.config_loader import CONFIG, SCHED, NOTIFY, FOCUS_TICKERS, SP100
from app.data_ingest import DataIngest
from app.data_stream import LiveStream
from app.model_evaluator import ModelEvaluator
from app.model_trainer import ModelTrainer
from app.monitor import SystemMonitor
from app.notify import discord_embed
from app.storage import DB
from app.trade_executor import TradeExecutor
from app.tests.validate_system import run_validation_and_report


logger = logging.getLogger("Orchestrator")


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger.info("📊 Starting orchestrator pipeline...")

    if not run_validation_and_report():
        message = "❌ Preflight failed — see logs for details"
        logger.error(message)
        discord_embed(
            "❌ Critical Failure",
            "Preflight validation failed. Inspect logs and resolve issues.",
            status="error",
            mention_role_id=NOTIFY.get("mention_role_id"),
        )
        print(message)
        raise SystemExit(1)

    db = DB()
    logger.info("🗄️ Connected to PostgreSQL database.")

    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"

    trainer = ModelTrainer(db, models_dir)
    evaluator = ModelEvaluator(db, models_dir)
    executor = TradeExecutor(db, models_dir, focus=FOCUS_TICKERS)
    ingest = DataIngest(db, CONFIG)

    key = os.getenv("ALPACA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    monitor = SystemMonitor(key, secret)

    logger.info("✅ All subsystems initialized.")
    logger.info("🧠 Model Trainer: ready")
    logger.info("💹 Trade Executor: %s mode", "paper" if executor.paper else "live")
    logger.info("🫀 Heartbeat active (%ss)", SCHED.get("heartbeat_seconds", 900))

    discord_embed(
        "✅ Trader Online",
        f"Realtime: {len(FOCUS_TICKERS)} focus tickers | Continuous operation engaged",
        status="ok",
        mention_role_id=NOTIFY.get("mention_role_id"),
    )

    stop_event = asyncio.Event()

    live_stream = LiveStream(key, secret, FOCUS_TICKERS) if SCHED.get("focus_stream", True) else None

    tasks = []

    if live_stream is not None:
        tasks.append(asyncio.create_task(_focus_stream_loop(live_stream, monitor, stop_event), name="focus_stream"))

    tasks.extend([
        asyncio.create_task(_sp100_loop(ingest, monitor, stop_event), name="sp100_loop"),
        asyncio.create_task(_news_loop(ingest, monitor, stop_event), name="news_loop"),
        asyncio.create_task(_fundamentals_loop(ingest, monitor, stop_event), name="fundamentals_loop"),
        asyncio.create_task(_model_training_loop(trainer, evaluator, executor, monitor, stop_event), name="model_training"),
        asyncio.create_task(_model_evaluation_loop(evaluator, monitor, stop_event), name="model_evaluation"),
        asyncio.create_task(_trade_execution_loop(executor, monitor, stop_event), name="trade_execution"),
        asyncio.create_task(_heartbeat_loop(monitor, stop_event), name="heartbeat"),
    ])

    logger.info("🔁 Continuous operation engaged...")

    def _request_shutdown() -> None:
        if stop_event.is_set():
            return
        logger.info("🛑 Shutdown signal received.")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, _request_shutdown)

    try:
        await stop_event.wait()
    except asyncio.CancelledError:  # pragma: no cover
        pass
    finally:
        logger.info("🧹 Commencing graceful shutdown...")
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
        discord_embed(
            "🛑 Bot Stopped",
            "Orchestrator shut down cleanly.",
            status="info",
            mention_role_id=NOTIFY.get("mention_role_id"),
        )
        with suppress(Exception):
            db.close()


async def _sp100_loop(ingest: DataIngest, monitor: SystemMonitor, stop_event: asyncio.Event) -> None:
    interval = SCHED.get("sp100_update_seconds", 300)
    while not stop_event.is_set():
        start = time.time()
        try:
            await asyncio.to_thread(ingest.pulse_yfinance_bars, SP100, "5m")
            elapsed = time.time() - start
            logger.info("[Ingest] ✅ SP100 updated (%.2fs)", elapsed)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[Ingest] ⚠️ SP100 update failed: %s", exc)
            monitor.record_connection_failure()
        await _sleep_with_stop(stop_event, start, interval)


async def _news_loop(ingest: DataIngest, monitor: SystemMonitor, stop_event: asyncio.Event) -> None:
    interval = SCHED.get("news_poll_seconds", 300)
    while not stop_event.is_set():
        start = time.time()
        try:
            await asyncio.to_thread(ingest.pulse_finnhub_news, SP100)
            elapsed = time.time() - start
            logger.info("[Ingest] 🗞️ News pulse complete (%.2fs)", elapsed)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[Ingest] ⚠️ News pulse failed: %s", exc)
            monitor.record_connection_failure()
        await _sleep_with_stop(stop_event, start, interval)


async def _fundamentals_loop(ingest: DataIngest, monitor: SystemMonitor, stop_event: asyncio.Event) -> None:
    interval = SCHED.get("fundamentals_poll_seconds", 900)
    while not stop_event.is_set():
        start = time.time()
        try:
            await asyncio.to_thread(ingest.pulse_fmp_fundamentals, SP100)
            elapsed = time.time() - start
            logger.info("[Ingest] 📊 Fundamentals refreshed (%.2fs)", elapsed)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[Ingest] ⚠️ Fundamentals pulse failed: %s", exc)
            monitor.record_connection_failure()
        await _sleep_with_stop(stop_event, start, interval)


async def _model_training_loop(
    trainer: ModelTrainer,
    evaluator: ModelEvaluator,
    executor: TradeExecutor,
    monitor: SystemMonitor,
    stop_event: asyncio.Event,
) -> None:
    interval = SCHED.get("model_train_seconds", 1800)
    while not stop_event.is_set():
        start = time.time()
        if monitor.halted:
            logger.info("[Model] ⏸️ Training paused (halt active)")
        else:
            try:
                result = await trainer.train()
                if result:
                    model_path, accuracy = result
                    monitor.update_metrics(model_accuracy=accuracy)
                    logger.info("[Model] 🧠 Training complete (accuracy %.3f) [%s]", accuracy, model_path.name)
                    try:
                        metrics = await evaluator.evaluate()
                        if metrics:
                            monitor.update_metrics(
                                eval_accuracy=metrics.get("accuracy"),
                                eval_f1=metrics.get("f1"),
                            )
                            logger.info(
                                "[Model] 📊 Evaluation metrics: acc=%.3f f1=%.3f",
                                metrics.get("accuracy", 0.0),
                                metrics.get("f1", 0.0),
                            )
                    except Exception as eval_exc:  # noqa: BLE001
                        logger.exception("[Model] ⚠️ Evaluation error post-training: %s", eval_exc)
                        monitor.record_connection_failure()
                    try:
                        executed = await executor.run()
                        if executed:
                            for trade in executed:
                                logger.info(
                                    "[Trade] 💹 %s BUY qty=%s @ %.2f (prob=%.2f)",
                                    trade["ticker"],
                                    trade["qty"],
                                    trade["price"],
                                    trade["probability"],
                                )
                    except Exception as trade_exc:  # noqa: BLE001
                        logger.exception("[Trade] ⚠️ Execution error post-training: %s", trade_exc)
                        monitor.record_connection_failure()
            except Exception as exc:  # noqa: BLE001
                logger.exception("[Model] ⚠️ Training error: %s", exc)
                monitor.record_connection_failure()
        await _sleep_with_stop(stop_event, start, interval)


async def _model_evaluation_loop(evaluator: ModelEvaluator, monitor: SystemMonitor, stop_event: asyncio.Event) -> None:
    interval = SCHED.get("model_eval_seconds", 3600)
    while not stop_event.is_set():
        start = time.time()
        if monitor.halted:
            logger.info("[Model] ⏸️ Evaluation paused (halt active)")
        else:
            try:
                metrics = await evaluator.evaluate()
                if metrics:
                    monitor.update_metrics(eval_accuracy=metrics.get("accuracy"), eval_f1=metrics.get("f1"))
                    logger.info("[Model] 📊 Evaluation metrics: acc=%.3f f1=%.3f", metrics.get("accuracy", 0.0), metrics.get("f1", 0.0))
            except Exception as exc:  # noqa: BLE001
                logger.exception("[Model] ⚠️ Evaluation error: %s", exc)
                monitor.record_connection_failure()
        await _sleep_with_stop(stop_event, start, interval)


async def _trade_execution_loop(executor: TradeExecutor, monitor: SystemMonitor, stop_event: asyncio.Event) -> None:
    interval = SCHED.get("trade_exec_seconds", 60)
    while not stop_event.is_set():
        start = time.time()
        if monitor.halted:
            logger.info("[Trade] ⏸️ Trading paused (halt active)")
        else:
            try:
                executed = await executor.run()
                if executed:
                    for trade in executed:
                        logger.info("[Trade] 💹 %s BUY qty=%s @ %.2f (prob=%.2f)", trade["ticker"], trade["qty"], trade["price"], trade["probability"])
            except Exception as exc:  # noqa: BLE001
                logger.exception("[Trade] ⚠️ Execution error: %s", exc)
                monitor.record_connection_failure()
        await _sleep_with_stop(stop_event, start, interval)


async def _heartbeat_loop(monitor: SystemMonitor, stop_event: asyncio.Event) -> None:
    interval = SCHED.get("heartbeat_seconds", 900)
    while not stop_event.is_set():
        start = time.time()
        try:
            await monitor.heartbeat()
            if monitor.try_auto_restart():
                logger.info("[Monitor] ♻️ Auto-restart triggered")
        except Exception as exc:  # noqa: BLE001
            logger.exception("[Monitor] ⚠️ Heartbeat error: %s", exc)
        await _sleep_with_stop(stop_event, start, interval)


async def _focus_stream_loop(stream: LiveStream, monitor: SystemMonitor, stop_event: asyncio.Event) -> None:
    interval = SCHED.get("focus_stream_seconds", 15)
    while not stop_event.is_set():
        start = time.time()
        if not stream.is_running():
            try:
                await stream.start()
                logger.info("[Stream] 📡 Focus stream active (%d symbols)", len(stream.tickers))
            except AttributeError:
                logger.warning("[WebSocket] ⚠️ Ignored 'NoneType.sock' from Alpaca client.")
            except Exception as exc:  # noqa: BLE001
                logger.exception("[Stream] ⚠️ Stream start failed: %s", exc)
                monitor.record_connection_failure()
        await _sleep_with_stop(stop_event, start, interval)
    with suppress(Exception):
        await stream.stop()


async def _sleep_with_stop(stop_event: asyncio.Event, start: float, interval: float) -> None:
    remaining = max(0.0, interval - (time.time() - start))
    if remaining <= 0:
        return
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=remaining)
    except asyncio.TimeoutError:
        return


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Orchestrator interrupted by user")
