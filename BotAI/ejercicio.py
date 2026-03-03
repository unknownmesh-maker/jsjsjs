"""Desarrollar un Bot de Discord que utiliza clasificación de imágenes con IA"""

import discord
from discord.ext import commands
import asyncio
import aiohttp
import os
from pathlib import Path
from PIL import Image
from io import BytesIO
from transformers import pipeline
import torch

# Configuración
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

# Carpeta para almacenar imágenes
IMAGES_FOLDER = "uploaded_images"
Path(IMAGES_FOLDER).mkdir(exist_ok=True)

# Inicializar el modelo de clasificación de imágenes
print("[INIT] Cargando modelo de clasificación de imágenes...")
try:
    image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
    print("[SUCCESS] Modelo cargado exitosamente")
except Exception as e:
    print(f"[ERROR] Error al cargar el modelo: {e}")
    image_classifier = None

@bot.event
async def on_ready():
    print(f'[READY] Bot conectado como {bot.user}')

@bot.command()
async def classify(ctx):
    """Clasifica una imagen usando IA"""
    
    if not ctx.message.attachments:
        embed = discord.Embed(
            title="❌ Error",
            description="Por favor adjunta una imagen al comando.",
            color=discord.Color.red()
        )
        await ctx.send(embed=embed)
        return
    
    if image_classifier is None:
        embed = discord.Embed(
            title="❌ Error",
            description="El modelo de IA no está disponible.",
            color=discord.Color.red()
        )
        await ctx.send(embed=embed)
        return
    
    attachment = ctx.message.attachments[0]
    
    # Validar que sea una imagen
    valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp')
    if not attachment.filename.lower().endswith(valid_extensions):
        embed = discord.Embed(
            title="❌ Formato inválido",
            description="El archivo debe ser una imagen válida (.png, .jpg, .jpeg, .gif, .webp, .bmp)",
            color=discord.Color.red()
        )
        await ctx.send(embed=embed)
        return
    
    # Mostrar estado de procesamiento
    processing_msg = await ctx.send("⏳ Procesando imagen... Por favor espera.")
    
    try:
        # Descargar la imagen
        image_data = await attachment.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        # Guardar la imagen
        file_path = os.path.join(IMAGES_FOLDER, attachment.filename)
        image.save(file_path)
        
        # Ejecutar clasificación en un executor para no bloquear el bot
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, image_classifier, image)
        
        # Crear embed con los resultados
        embed = discord.Embed(
            title="🤖 Resultados de Clasificación",
            description=f"**Imagen:** {attachment.filename}",
            color=discord.Color.green()
        )
        
        # Agregar top 5 resultados
        for i, result in enumerate(results[:5], 1):
            label = result['label']
            score = result['score']
            percentage = f"{score*100:.2f}%"
            embed.add_field(
                name=f"#{i} {label}",
                value=f"Confianza: {percentage}",
                inline=False
            )
        
        embed.set_thumbnail(url=attachment.url)
        embed.set_footer(text=f"Tamaño: {len(image_data)} bytes")
        
        # Eliminar mensaje de procesamiento y enviar resultados
        await processing_msg.delete()
        await ctx.send(embed=embed)
        
        print(f"[INFO] Imagen clasificada: {attachment.filename}")
        
    except Exception as e:
        await processing_msg.delete()
        embed = discord.Embed(
            title="❌ Error al procesar",
            description=f"```{str(e)}```",
            color=discord.Color.red()
        )
        await ctx.send(embed=embed)
        print(f"[ERROR] {str(e)}")

@bot.command()
async def list_images(ctx):
    """Lista todas las imágenes guardadas"""
    try:
        images = os.listdir(IMAGES_FOLDER)
        if not images:
            embed = discord.Embed(
                title="📁 Imágenes",
                description="No hay imágenes guardadas.",
                color=discord.Color.blue()
            )
            await ctx.send(embed=embed)
            return
        
        embed = discord.Embed(
            title="📸 Imágenes Guardadas",
            description="\n".join([f"• {img}" for img in images[:20]]),
            color=discord.Color.blue()
        )
        embed.set_footer(text=f"Total: {len(images)} imágenes")
        await ctx.send(embed=embed)
        
    except Exception as e:
        embed = discord.Embed(
            title="❌ Error",
            description=f"```{str(e)}```",
            color=discord.Color.red()
        )
        await ctx.send(embed=embed)

@bot.command()
async def info(ctx):
    """Muestra información sobre el bot"""
    embed = discord.Embed(
        title="🤖 Bot de Clasificación de Imágenes",
        description="Un bot que utiliza IA para clasificar imágenes",
        color=discord.Color.purple()
    )
    embed.add_field(name="Comandos disponibles:", value="""
    `!classify` - Clasifica una imagen adjunta
    `!list_images` - Lista imágenes guardadas
    `!info` - Muestra esta información
    """, inline=False)
    embed.add_field(name="Modelo:", value="Vision Transformer (google/vit-base-patch16-224)", inline=False)
    embed.add_field(name="Framework:", value="Hugging Face Transformers", inline=False)
    await ctx.send(embed=embed)


