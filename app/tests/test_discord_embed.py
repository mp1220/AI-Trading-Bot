import os

from app.utils.discord_utils import discord_embed


def run_discord_embed_tests():
    print("🧪 Testing Discord embed utility...")

    long_message = "A" * 2100  # exceed 2000 chars
    result1 = discord_embed(
        "✅ Test 1 – Length Handling",
        description=long_message,
        color=0x3498DB,
        channel="backend",
    )
    print("Test 1 result:", result1)

    result2 = discord_embed(
        "✅ Test 2 – Channel Mapping",
        description="Testing message routing via channel='market-updates'.",
        color=0x2ECC71,
        channel="market-updates",
    )
    print("Test 2 result:", result2)

    result3 = discord_embed(
        "✅ Test 3 – Manual Webhook",
        description="Direct webhook send test.",
        color=0xE67E22,
        webhook_url=os.getenv("DISCORD_WEBHOOK_TRADES"),
    )
    print("Test 3 result:", result3)


if __name__ == "__main__":
    run_discord_embed_tests()
