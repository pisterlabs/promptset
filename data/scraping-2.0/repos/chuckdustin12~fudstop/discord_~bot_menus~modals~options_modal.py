# At the top of the file.
import disnake
from disnake.ext import commands
from disnake import TextInputStyle
import aiohttp
import os
import asyncio
import openai
# Subclassing the modal.
class OptionModalView(disnake.ui.View):
    def __init__(self):
        super().__init__(timeout=None)



        self.add_item(QuoteSelect())
        self.add_item(GreekSelect())
        self.add_item(TradeSelect())
        self.add_item(MarketDataSelect())
        button = disnake.ui.Button(style=disnake.ButtonStyle.blurple, emoji=f"<a:_:1044503166433099856>", row=4)
        self.add_item(button)



class GreekSelect(disnake.ui.Select):

    def __init__(self):
        super().__init__( 
            placeholder=f"Filter Greeks -->",
            custom_id='filtergreeks',
            min_values=1,
            max_values=5,
            options = [
                    disnake.SelectOption(
                        label="Gamma",
                        description="Measures Delta's change per $1 increase in stock."
                    ),
                    disnake.SelectOption(
                        label="Vega",
                        description="Option price sensitivity to volatility."
                    ),
                    disnake.SelectOption(
                        label="Delta",
                        description="Option price change per $1 move in stock."
                    ),
                    disnake.SelectOption(
                        label="IV",
                        description="Estimate of future security price volatility."
                    ),
                    disnake.SelectOption(
                        label="Theta",
                        description="Option price time decay rate per day."
                    )
])

    
    async def callback(self, interaction: disnake.MessageCommandInteraction):
        # Assuming you have a function to create and return a modal
        modal = self.create_modal_based_on_selection(self.values)
        await interaction.response.send_modal(modal)

    def create_modal_based_on_selection(self, selections):
        components = []
        
        for selection in selections:
            components.append(
            disnake.ui.TextInput(
                label=selection,
                custom_id=f'input_{selection.lower()}',
                style=disnake.TextInputStyle.short,
                placeholder=f'Enter {selection} value...',
            
            ))

        
        return GreeksModal(components)





class OptionsDataModal(disnake.ui.Modal):
    """
    Perform options related tasks
    
    """
    def __init__(self):
        # The details of the modal, and its components
        components = [
            disnake.ui.TextInput(
                label="Underlying Symbol",
                placeholder="e.g. AAPL",
                custom_id="symbol",
                style=TextInputStyle.paragraph,
                max_length=8,
            ),
            disnake.ui.TextInput(
                label="Strike Price",
                placeholder="e.g. 12.5 | 30",
                custom_id="strikeprice",
                style=TextInputStyle.paragraph,
                max_length=8,
            ),
            disnake.ui.TextInput(
                label="Call or Put?",
                placeholder="e.g. call | put",
                custom_id="callput",
                style=TextInputStyle.paragraph,
                max_length=4,
            ),

            disnake.ui.TextInput(
                label="Expiry Date",
                placeholder="e.g. 2023-12-15",
                custom_id="expiry",
                style=TextInputStyle.paragraph,
                max_length=10,
            ),

        ]
        super().__init__(title="Build your Symbol:", components=components)

    # The callback received when the user input is completed.
    async def callback(self, inter: disnake.ModalInteraction):
        await inter.response.defer(with_message=True, ephemeral=False)

        embed = disnake.Embed(title=f"Option Loaded - {inter.text_values.get('symbol')} | ${inter.text_values.get('strikeprice')} | {inter.text_values.get('callput')} | {inter.text_values.get('expiry')}", description=f"> # **Use the dropdown below to choose data**", color=disnake.Colour.dark_teal())
        await inter.edit_original_message(embed=embed, view=OptionModalView())


class GreeksModal(disnake.ui.Modal):
    """
    Perform options related tasks
    
    """
    def __init__(self, components):
        components = components
        # The details of the modal, and its components
        # components = [
        #     disnake.ui.TextInput(
        #         label="Enter a delta range -->",
        #         placeholder="e.g. delta >= 0.5",
        #         custom_id="delta",
        #         style=TextInputStyle.short,
        #         max_length=100,
        #         required=False
        #     ),
        #     disnake.ui.TextInput(
        #         label="Enter a gamma range -->",
        #         placeholder="e.g. >= 0.07",
        #         custom_id="gamma",
        #         style=TextInputStyle.short,
        #         max_length=100,
        #         required=False
        #     ),
        #     disnake.ui.TextInput(
        #         label="Enter a theta range -->",
        #         placeholder="e.g. <= 0.02",
        #         custom_id="theta",
        #         style=TextInputStyle.short,
        #         max_length=10,
        #         required=False
        #     ),

        #     disnake.ui.TextInput(
        #         label="Enter a vega range -->",
        #         placeholder='e.g. <= 0.05',
        #         custom_id="vega",
        #         style=TextInputStyle.short,
        #         max_length=10,
        #         required=False
        #     ),

        #     disnake.ui.TextInput(
        #         label="Enter an IV range..",
        #         placeholder="e.g. >= 25",
        #         custom_id="iv",
        #         style=TextInputStyle.short,
        #         max_length=7,
        #     ),

        # ]
        super().__init__(title="Filter By Greeks:", components=components)

    # The callback received when the user input is completed.
    async def callback(self, inter: disnake.ModalInteraction):
        await inter.response.defer(with_message=True)

        await inter.edit_original_message('> # Chain Modal worked! Prepare autism!')



class QuoteSelect(disnake.ui.Select):

    def __init__(self):
        options = [
            disnake.SelectOption(
                label="Ask",
                description="Enter the asking price."
            ),
            disnake.SelectOption(
                label="Ask Size",
                description="Enter the size of the ask."
            ),
            disnake.SelectOption(
                label="Bid",
                description="Enter the bidding price."
            ),
            disnake.SelectOption(
                label="Bid Size",
                description="Enter the size of the bid."
            ),
            disnake.SelectOption(
                label="Midpoint",
                description="Enter the midpoint value."
            )
        ]
        super().__init__(
            placeholder="Filter Quotes -->",
            custom_id='filterquotes',
            min_values=1,
            max_values=5,
            options=options
        )
    
    async def callback(self, interaction: disnake.MessageCommandInteraction):
        modal = self.create_modal_based_on_selection(self.values)
        await interaction.response.send_modal(modal)

    def create_modal_based_on_selection(self, selections):
        components = []
        
        for selection in selections:
            components.append(
                disnake.ui.TextInput(
                    label=selection,
                    custom_id=f'input_{selection.lower().replace(" ", "_")}',
                    style=disnake.TextInputStyle.short,
                    placeholder=f'Enter {selection} value...',
                )
            )
        
        return QuotesModal(components)


class QuotesModal(disnake.ui.Modal):
    """
    Perform options related tasks
    
    """
    def __init__(self, components):
        components = components


        super().__init__(title="Filter the quotes:", components=components)

    # The callback received when the user input is completed.
    async def callback(self, inter: disnake.ModalInteraction):
        await inter.response.defer(with_message=True)

        await inter.edit_original_message('> # Quotes chain - Worked!')



class TradeSelect(disnake.ui.Select):

    def __init__(self):
        options = [
            disnake.SelectOption(
                label="Conditions",
                description="Trade conditions."
            ),
            disnake.SelectOption(
                label="Exchange",
                description="Exchange identifier."
            ),
            disnake.SelectOption(
                label="Price",
                description="Trade price."
            ),
            disnake.SelectOption(
                label="Timestamp",
                description="SIP timestamp."
            ),
            disnake.SelectOption(
                label="Size",
                description="Trade size."
            )
        ]
        super().__init__(
            placeholder="Filter Trades -->",
            custom_id='filtertrades',
            min_values=1,
            max_values=5,
            options=options
        )


    async def callback(self, interaction: disnake.MessageCommandInteraction):
            modal = self.create_modal_based_on_selection(self.values)
            await interaction.response.send_modal(modal)

    def create_modal_based_on_selection(self, selections):
        components = []
        
        for selection in selections:
            components.append(
                disnake.ui.TextInput(
                    label=selection,
                    custom_id=f'input_{selection.lower().replace(" ", "_")}',
                    style=disnake.TextInputStyle.short,
                    placeholder=f'Enter {selection} value...',
                )
            )
        
        return TradeModal(components)




class TradeModal(disnake.ui.Modal):
    """
    Perform options related tasks
    
    """
    def __init__(self, components):
        components = components


        super().__init__(title="Filter the trades:", components=components)

    # The callback received when the user input is completed.
    async def callback(self, inter: disnake.ModalInteraction):
        await inter.response.defer(with_message=True)

        await inter.edit_original_message('> # Trades chain - Worked!')



class MarketDataSelect(disnake.ui.Select):

    def __init__(self):
        options = [
            disnake.SelectOption(
                label="Change Percent",
                description="Change in price by percentage."
            ),
            disnake.SelectOption(
                label="Open Interest",
                description="Total outstanding contracts."
            ),
            disnake.SelectOption(
                label="Volume",
                description="Volume of trades."
            ),
            disnake.SelectOption(
                label="VWAP",
                description="Volume Weighted Average Price."
            ),
            disnake.SelectOption(
                label="Underlying Price",
                description="Price of the underlying asset."
            )
        ]
        super().__init__(
            placeholder="Select Market Data -->",
            custom_id='filtermarketdata',
            min_values=1,
            max_values=5,
            options=options
        )

    async def callback(self, interaction: disnake.MessageCommandInteraction):
        modal = self.create_modal_based_on_selection(self.values)
        await interaction.response.send_modal(modal)

    def create_modal_based_on_selection(self, selections):
        components = []
        
        for selection in selections:
            component_custom_id = f'input_{selection.lower().replace(" ", "_")}'
            components.append(
                disnake.ui.TextInput(
                    label=selection,
                    custom_id=component_custom_id,
                    style=disnake.TextInputStyle.short,
                    placeholder=f'Enter {selection}...',
                )
            )
        
        return MarketDataModal(components)
    



class MarketDataModal(disnake.ui.Modal):
    """
    Perform options related tasks
    
    """
    def __init__(self, components):
        components = components


        super().__init__(title="Filter Acivity:", components=components)

    # The callback received when the user input is completed.
    async def callback(self, inter: disnake.ModalInteraction):
        await inter.response.defer(with_message=True)

        await inter.edit_original_message('> # Activity chain - Worked!')