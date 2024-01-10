import interactions

bot = interactions.Client(token="MTEzNzE5OTg4OTk5OTIwODQ0OA.GU86xE.W6eGJXDGEUT5PIfzxdyb-cikSR8YW_XOmRkzQk")

@interactions.slash_command(
    name="askDitka",
    description="Ask Coach Ditka anything!",
)
async def askDitka(ctx: interactions.CommandContext):
    # Replace this part with the logic to get a response from OpenAI
    response_from_openai = "Coach Ditka says, 'Keep working hard!'"
    
    print(f"Received command from {ctx.author}: {ctx.command}")
    print(f"Responding with: {response_from_openai}")

    await ctx.send(response_from_openai)

@interactions.slash_command(
    name="ping",
    description="Ping command",
)
async def ping(ctx: interactions.CommandContext):
    print(f"Received ping from {ctx.author}")
    await ctx.send("Pong!")

print("Bot is running...")
bot.start()


