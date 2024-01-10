import discord
from discord.ext import commands

import openai

import generator
import main
from main import logger
import views

import utils
import datetime

import sympy as sp


class Math(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @discord.slash_command(name='math', description='Solve math problems using CAS!')
    @discord.option("prompt", description="Enter a prompt.")
    @discord.option("imaginary", description="Allow imaginary numbers?", choices=[True, False])
    async def math_command(self, interaction: discord.Interaction,
                           prompt: str,
                           imaginary: bool = False):
        await interaction.response.defer()
        start_time = datetime.datetime.now()

        response = "Invalid equation"
        if prompt.__contains__('='):
            left, right = prompt.split("=")
            left, right = utils.rewrite_implicit_multiplication(left), utils.rewrite_implicit_multiplication(right)
            try:
                left, right = sp.sympify(left, evaluate=True, convert_xor=True), sp.sympify(right, evaluate=True,
                                                                                            convert_xor=True)
                solution = sp.solve(sp.Eq(left, right), set=True)
                response = ""
                logger.info(solution)
                logger.info(f"Solution Vars: {solution[0]} | Sets: {solution[1]}")
                for i in range(len(solution[0])):
                    response += str(solution[0][i]) + "="
                    for j in solution[1]:
                        for k in j:
                            # Check if k is a real number
                            if k.is_real or imaginary:
                                response += str(k) + ", "
                    response += "\n"
            except sp.SympifyError as e:
                response = "Invalid equation - " + str(e)
        else:
            solution = sp.sympify(utils.rewrite_implicit_multiplication(prompt), evaluate=True, convert_xor=True)
            response = solution

        embed = utils.create_embed(title="CAS Response", description=f"`{response}`")
        embed.add_field(name="Prompt", value=f"`{prompt}`")
        embed.set_footer(text=f"MathGPT • {(datetime.datetime.now() - start_time).total_seconds().__round__(2)}s",
                         icon_url=self.bot.user.display_avatar.url)

        await interaction.followup.send(embed=embed, view=views.MathView())

    @discord.slash_command(name='math-gpt', description='Solve math problems using GPT-3 and CAS')
    @discord.option("prompt", description="Enter a prompt.")
    @discord.option("imaginary", description="Allow imaginary numbers?", choices=[True, False])
    async def math_gpt_command(self, interaction: discord.Interaction,
                               prompt: str,
                               imaginary: bool = False):
        await interaction.response.defer()
        start_time = datetime.datetime.now()

        # run the prompt through the recognition model
        completion = openai.Completion.create(engine="babbage:ft-mathgpt:recognition-2023-03-02-07-27-11", prompt=prompt + "\n\n###\n\n", max_tokens=60, temperature=0.0, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, stop=["###"])
        # remove the "###" from the end of the response and split it into two variables with the "|" indice
        completion = completion["choices"][0]["text"]
        print(completion)
        completion_l = completion.split("|")

        # get the problem_type and parameters from the response
        problem_type = completion_l[len(completion_l) - 1]

        # remove the problem_type from the parameters
        parameters = completion_l[0:len(completion_l) - 1]

        problem_type = problem_type.strip()

        # identify the problem_type as a Problemproblem_type
        problem_type = generator.ProblemType[problem_type.upper()]

        response = ""

        # solve the problem
        try:
            solution = problem_type.solver(parameters)
        except sp.SympifyError as e:
            response = "Invalid equation - " + str(e)
        if response == "":
            # depending on the problem_type of solution, produce a different output
            if isinstance(solution, sp.Expr):
                response = str(solution)
            elif isinstance(solution, list):
                response = ", ".join(solution)
            elif isinstance(solution, tuple):
                print(f"Solution Vars: {solution[0]} | Sets: {solution[1]}")
                variables = solution[0]
                sets = solution[1]

                for i in range(len(variables)):
                    print("i: " + str(i))
                    print(variables[i])
                    response += (str(variables[i]) + "=")
                    for j in sets:
                        for k in j:
                            # Check if k is a real number
                            if k.is_real or imaginary:
                                response += str(k) + ", "
                    response += "\n"
                # response = str(solution)
            elif isinstance(solution, bool):
                response = "True" if solution else "False"
            else:
                response = "Invalid equation - " + str(type(solution))

        embed = utils.create_embed(title="CAS Response", description=f"`{response}`")
        embed.add_field(name="Prompt", value=f"`{prompt}`")
        embed.add_field(name="problem_type", value=f"`{problem_type}`")
        embed.set_footer(text=f"MathGPT • {(datetime.datetime.now() - start_time).total_seconds().__round__(2)}s",
                         icon_url=self.bot.user.display_avatar.url)

        await interaction.followup.send(embed=embed, view=views.MathView())


def setup(bot):
    bot.add_cog(Math(bot))
    main.logger.info("Loaded Math cog")
