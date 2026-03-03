import discord
from discord.ext import commands
from model import get_class
import os
import random
import requests

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command()
async def hello(ctx):
    await ctx.send(f'Hi! I am a bot {bot.user}!')

@bot.command()
async def heh(ctx, count_heh = 5):
    await ctx.send("he" * count_heh)

""""Lo que tenemos que hacer es implementar la lógica que permitirá al usuario subir una imagen. A continuación, el bot debe guardar esta imagen para su posterior procesamiento."""

"""
@bot.command()
async def upload(ctx):
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]
        await attachment.save(attachment.filename)
        await ctx.send(f'Image {attachment.filename} has been saved!')
    else:
        await ctx.send('Please attach an image to upload.') """
        

@bot.command()
async def check(ctx):
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            file_name = attachment.filename
            file_url = attachment.url
            await attachment.save(f"./{attachment.filename}")
            await ctx.send(get_class(model_path="./keras_model.h5", labels_path="labels.txt", image_path=f"./{attachment.filename}"))
    else:
        await ctx.send('Please attach an image to check.')