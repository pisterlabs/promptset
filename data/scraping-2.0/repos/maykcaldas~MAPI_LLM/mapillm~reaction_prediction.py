import os
import re
from langchain.agents import Tool, tool
# from mp_api.client import MPRester
from pymatgen.ext.matproj import MPRester
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.enumerators.basic import BasicEnumerator

class SynthesisReactions:
    def __init__(self, temp=900, stabl=0.025, exclusive_precursors=False, exclusive_targets=False):
        self.temp = temp
        self.stabl = stabl
        self.exclusive_precursors = exclusive_precursors
        self.exclusive_targets = exclusive_targets
        
    def _split_string(self, s):
        if isinstance(s, list):
            s = "".join(s)
        parts = re.findall('[a-z]+|[A-Z][a-z]*', s)
        letters_only = [re.sub(r'\d+', '', part) for part in parts]
        unique_letters = list(set(letters_only))
        result = "-".join(unique_letters)
        return result

    def _get_rxn_from_precursor(self, precursors_formulas):
        prec = precursors_formulas.split(',') if "," in precursors_formulas else precursors_formulas

        with MPRester(os.getenv("MAPI_API_KEY")) as mpr:  
            entries = mpr.get_entries_in_chemsys(self._split_string(prec))

        gibbs_entries = GibbsEntrySet.from_computed_entries(entries, self.temp)
        filtered_entries = gibbs_entries.filter_by_stability(self.stabl)

        prec = [prec] if isinstance(prec, str) else prec
        be = BasicEnumerator(precursors=prec, exclusive_precursors=self.exclusive_precursors)
        rxns = be.enumerate(filtered_entries)
        try:
            rxn_choice = next(iter(rxns))
            return str(rxn_choice)
        except: 
            return "Error: No reactions found."

    def _get_rxn_from_target(self, targets_formulas):
        targets = targets_formulas.split(',') if "," in targets_formulas else targets_formulas

        with MPRester(os.getenv("MAPI_API_KEY")) as mpr:  
            entries = mpr.get_entries_in_chemsys(self._split_string(targets))

        gibbs_entries = GibbsEntrySet.from_computed_entries(entries, self.temp)
        filtered_entries = gibbs_entries.filter_by_stability(self.stabl)

        targets = [targets] if isinstance(targets, str) else targets

        be = BasicEnumerator(targets=targets, exclusive_targets=self.exclusive_targets)
        rxns = be.enumerate(filtered_entries)
        try:
            rxn_choice = next(iter(rxns))
            return str(rxn_choice)
        except: 
            return "Error: No reactions found."

    def _break_equation(self, equation):
        pattern = r'(\d*\.?\d*\s*[A-Za-z]+\d*|\+|\->)'
        pieces = re.findall(pattern, equation)
        equation_pieces = []
        current_piece = ''
        for piece in pieces:
            if piece == '+' or piece == '->':
                equation_pieces.append(current_piece.strip())
                equation_pieces.append(piece)
                current_piece = ''
            else:
                current_piece += piece + ' '
        equation_pieces.append(current_piece.strip())
        return equation_pieces

    def _convert_equation_pieces(self, equation_pieces):
        if '+' in equation_pieces:
            equation_pieces = [piece if piece != '+' else 'with' for piece in equation_pieces]
            equation_pieces = [piece if piece != '->' else 'to yield' for piece in equation_pieces]
        else:
            equation_pieces = [piece if piece != '->' else 'yields' for piece in equation_pieces]
        return equation_pieces

    def _split_equation_pieces(self, equation_pieces):
        new_pieces = []
        for piece in equation_pieces:
            if piece in ["with", "to yield", "yields"]:
                new_pieces.append(piece)
            else:
                if re.match(r'^\d*\.\d+|\d+', piece):
                    number_match = re.match(r'^\d*\.\d+|\d+', piece)
                    number = number_match.group(0)
                    rest = piece[len(number):]
                    new_pieces.append(number)
                    new_pieces.append(rest)
                else:
                    new_pieces.append("1")
                    new_pieces.append(piece)
        return new_pieces

    def _modify_mols(self, equation_pieces):
        for i, piece in enumerate(equation_pieces):
            if piece.replace('.', '', 1).isdigit():
                equation_pieces[i] = f"{piece} mols"
        return equation_pieces

    def _combine_equation_pieces(self, equation_pieces):
        if 'with' in equation_pieces:
            equation_pieces.insert(0, 'mix')
        combined_string = ' '.join(equation_pieces)
        return combined_string

    def _process_equation(self, equation):
        equation_pieces = self._break_equation(equation)
        converted_pieces = self._convert_equation_pieces(equation_pieces)
        split_pieces = self._split_equation_pieces(converted_pieces)
        modified_pieces = self._modify_mols(split_pieces)
        combined_string = self._combine_equation_pieces(modified_pieces)
        return combined_string
    
    def get_reaction(self, input_string):
        input_parts = input_string.split(',', 1)
        if len(input_parts) != 2:
            raise ValueError("Invalid input format. Expected 'precursor' or 'target', followed by a comma, and then the list of formulas separated by a comma.")
        
        mode, formulas = input_parts
        mode = mode.lower().strip()

        if mode == "precursor":
            reaction = self._get_rxn_from_precursor(formulas)
        elif mode == "target":
            reaction = self._get_rxn_from_target(formulas)
        else:
            raise ValueError("Invalid mode. Expected 'precursor' or 'target'.")
        processed_reaction = self._process_equation(reaction)
        return processed_reaction

    def get_tools(self):
        return [
            Tool(
                name = "Get a synthesis reaction for a material",
                func = self.get_reaction,
                description = (
                "This function is useful for suggesting a synthesis reaction for a material. "
                "Give this tool a string containing either precursor or target, then a comma, followed by the formulas separated by comma as input and returns a synthesis reaction."
                "The mode is used to determine if the input is a precursor or a target material. "
                )
        )]

