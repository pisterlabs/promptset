import discord
from discord.ext import commands
import traceback
from embed import color_Embed
import random
import asyncio
import openai
from .Login import usuarios_autenticados

# ****************************
# **    Lista de Comidas    **
# ****************************
comida_mexicana = [
    "Tacos",
    "Enchiladas",
    "Guacamole",
    "Tamales",
    "Chiles en nogada",
    "Pozole",
    "Quesadillas",
    "Sopes",
    "Ceviche",
    "Mole",
    "Churros",
    "Flan",
]

comida_rapida = [
    "Hamburguesa",
    "Papas fritas",
    "Pizza",
    "Hot dogs",
    "Pollo frito",
    "Sándwiches",
    "Tacos de pescado",
    "Nuggets de pollo",
    "Burritos",
    "Donas",
    "Nachos",
    "Milkshake",
]

comida_japonesa = [
    "Sushi",
    "Sashimi",
    "Ramen",
    "Tempura",
    "Teriyaki",
    "Gyoza",
    "Okonomiyaki",
    "Yakitori",
    "Udon",
    "Onigiri",
    "Matcha",
    "Dorayaki",
]

# **************************************************
# **    Comando para Lanzar una Moneda al Azar    **
# **************************************************
def Moneda(bot):
    @bot.command()
    async def moneda(ctx):
        if ctx.author.id in usuarios_autenticados:
            try:
                # Mostrando "Escribiendo..."
                async with ctx.typing():
                    await asyncio.sleep(3)  # Esperando 3 segundos simulando "Escribiendo..."

                # Obtener el color correspondiente al comando
                color_tuple = color_Embed["moneda"]
                color = discord.Colour.from_rgb(*color_tuple)

                resultado = random.choice(["cara", "cruz"])

                # Crear un Embed para el resultado de la moneda
                embed = discord.Embed(title="Lanzamiento de Moneda", color=color)
                embed.add_field(name="Resultado", value=resultado, inline=False)

                # Comprobar si el autor tiene una imagen de perfil
                if ctx.author.avatar:
                    embed.set_footer(text=f"Lanzado por {ctx.author.display_name}", icon_url=ctx.author.avatar.url)
                else:
                    embed.set_footer(text=f"Lanzado por {ctx.author.display_name}")

                # Agregar una imagen de una moneda (cara o cruz) al Embed
                if resultado == "cara":
                    embed.set_image(url="https://www.banxico.org.mx/multimedia/mon20_700anosMxTen_revPrensa.jpg")
                else:
                    embed.set_image(url="https://www.banxico.org.mx/multimedia/anv_mon20Marina_prensaNgo.png")

                await ctx.send(embed=embed)

            except Exception as e:
                # Mostrando un error en rojo
                color_error = discord.Colour.from_rgb(255, 0, 0)
                embed_error = discord.Embed(title="Error", color=color_error)
                embed_error.add_field(name="Error al tirar la moneda", value=str(e), inline=False)
                await ctx.send(embed=embed_error)
        else:
            await ctx.send("No tienes permiso para usar este comando protegido. Por favor, inicia sesión primero.")

# ***********************************************
# **    Comando para Jugar a la Ruleta Rusa    **
# ***********************************************
def Ruleta_Rusa(bot):
    @bot.command()
    async def ruleta(ctx):
        if ctx.author.id in usuarios_autenticados:
            try:
                # Mostrando "Escribiendo..."
                async with ctx.typing():
                    await asyncio.sleep(3)  # Esperando 3 segundos simulando "Escribiendo..."

                # Obtener el color correspondiente al comando
                color_tuple = color_Embed["ruleta"]
                color = discord.Colour.from_rgb(*color_tuple)

                chamber = random.randint(1, 6)
                trigger = random.randint(1, 6)

                # Crear un Embed para el juego de la ruleta rusa
                embed = discord.Embed(title="Ruleta Rusa", color=color)

                # Comprobar si el autor tiene una imagen de perfil
                if ctx.author.avatar:
                    embed.set_footer(text=f"Jugado por {ctx.author.display_name}", icon_url=ctx.author.avatar.url)
                else:
                    embed.set_footer(text=f"Jugado por {ctx.author.display_name}")

                if chamber == trigger:
                    embed.add_field(name="¡BANG!", value="Parece que no sobreviviste. 😵💥", inline=False)
                else:
                    embed.add_field(name="¡Click!", value="Sobreviviste por ahora. 😅🔫", inline=False)

                await ctx.send(embed=embed)

            except Exception as e:
                # Mostrando un error en rojo
                color_error = discord.Colour.from_rgb(255, 0, 0)
                embed_error = discord.Embed(title="Error", color=color_error)
                embed_error.add_field(name="Error, se trabó el revólver", value=str(e), inline=False)
                await ctx.send(embed=embed_error)
        else:
            await ctx.send("No tienes permiso para usar este comando protegido. Por favor, inicia sesión primero.")

# ***********************************
# **    Comando para dar Comida    **
# ***********************************
def Elegir_Comida(bot):
    @bot.command()
    async def comida(ctx):
        if ctx.author.id in usuarios_autenticados:
            try:
                # Mostrando "Escribiendo..."
                async with ctx.typing():
                    await asyncio.sleep(3)  # Esperando 3 segundos simulando "Escribiendo..."

                # Obtener el color correspondiente al comando
                color_tuple = color_Embed["comida"]
                color = discord.Colour.from_rgb(*color_tuple)

                # Elegir una categoría de comida al azar
                categoria = random.choice([comida_mexicana, comida_rapida, comida_japonesa])

                # Elegir una comida al azar de la categoría
                comida_elegida = random.choice(categoria)

                # Obtener el nombre de la categoría
                nombre_categoria = ""
                if categoria == comida_mexicana:
                    nombre_categoria = "Comida Mexicana"
                elif categoria == comida_rapida:
                    nombre_categoria = "Comida Rápida"
                elif categoria == comida_japonesa:
                    nombre_categoria = "Comida Japonesa"

                # Crear un Embed con el mensaje de comida y categoría
                embed = discord.Embed(title="Elección de Comida", color=color)
                embed.add_field(name=f"Hoy te toca comer algo de {nombre_categoria}:", value=f"**`{comida_elegida}`** para la cena", inline=False)

                # Comprobar si el autor tiene una imagen de perfil
                if ctx.author.avatar:
                    embed.set_footer(text=f"Solicitado por {ctx.author.display_name}", icon_url=ctx.author.avatar.url)
                else:
                    embed.set_footer(text=f"Solicitado por {ctx.author.display_name}")

                # Enviar el Embed como mensaje personal al usuario
                await ctx.author.send(embed=embed)

                # Crear un Embed para el mensaje en el servidor
                embed_server = discord.Embed(
                    title="Comida Enviada",
                    description=f"¡Hemos enviado tu comida a tu casa, {ctx.author.mention}! Revisa tus mensajes privados.",
                    color=color
                )
                await ctx.send(embed=embed_server)

            except Exception as e:
                # Mostrando un error en rojo
                color_error = discord.Colour.from_rgb(255, 0, 0)
                embed_error = discord.Embed(title="Error", color=color_error)
                embed_error.add_field(name="Ocurrió un error al elegir la comida", value=str(e), inline=False)
                await ctx.send(embed=embed_error)
        else:
            await ctx.send("No tienes permiso para usar este comando protegido. Por favor, inicia sesión primero.")

# **********************************************
# **    Comando para Implementar a ChatGPT    **
# **********************************************
def GPT(bot):
    @bot.command()
    async def chat(ctx, *, pregunta):
        if ctx.author.id in usuarios_autenticados:
            try:
                # Mostrando "Escribiendo..."
                async with ctx.typing():
                    await asyncio.sleep(2)  # Esperando 2 segundos simulando "Escribiendo..."

                response = openai.Completion.create(
                    engine="text-davinci-002",
                    prompt=pregunta,
                    max_tokens=150
                )
                respuesta = response.choices[0].text

                # Comprobar si el autor tiene una imagen de perfil
                if ctx.author.avatar:
                    await ctx.send(respuesta, content=f"Respuesta para {ctx.author.mention}:", embed=discord.Embed().set_thumbnail(url=ctx.author.avatar.url))
                else:
                    await ctx.send(respuesta)

            except Exception as e:
                # Mostrando un error en rojo
                color_error = discord.Colour.from_rgb(255, 0, 0)
                embed_error = discord.Embed(title="Error", color=color_error)
                embed_error.add_field(name="Ocurrió un error", value=str(e), inline=False)
                await ctx.send(embed=embed_error)
        else:
            await ctx.send("No tienes permiso para usar este comando protegido. Por favor, inicia sesión primero.")
