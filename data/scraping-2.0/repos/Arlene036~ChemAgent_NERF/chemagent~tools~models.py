from langchain.tools import BaseTool
from transformers import AutoTokenizer, T5ForConditionalGeneration
from .NERF.preprocess import molecule
from .NERF.model import *
from .NERF.dataset import TransformerDataset
from torch.utils.data import DataLoader
from rdkit import Chem
from .NERF.utils import result2mol
from rdkit import Chem
import openai
import os
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor

rdDepictor.SetPreferCoordGen(True)
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG
import rdkit

parser = argparse.ArgumentParser(description='')

check_point = ["epoch-98-loss-1.1171849480149887"]

parser.add_argument('--data_path', type=str, help='path of dataset', default='./')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size.256')
parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle the order of atoms')
parser.add_argument('--num_workers', type=int, default=4, help='num workers to generate data.')
parser.add_argument('--prefix', type=str, default='data',
                    help='data prefix')

parser.add_argument('--name', type=str, default='tmp',
                    help='model name, crucial for test and checkpoint initialization')
parser.add_argument('--vae', action='store_true', default=True, help='use vae')
parser.add_argument('--depth', type=int, default=6, help='depth')
parser.add_argument('--dim', type=int, default=192, help='dim')

parser.add_argument('--save_path', type=str, default='../tools/NERF/CKPT/no_reactant_mask/', help='path of save prefix')
parser.add_argument('--train', action='store_true', default=False, help='do training.')
parser.add_argument('--save', action='store_true', default=True, help='Save model.')
parser.add_argument('--eval', action='store_true', default=True, help='eval model.')
parser.add_argument('--test', action='store_true', default=True, help='test model.')
parser.add_argument('--recon', action='store_true', default=False, help='test reconstruction only.')

parser.add_argument('--seed', type=int, default=2019, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train. 200')
parser.add_argument('--local_rank', default='cpu', help='rank')
parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
parser.add_argument('--temperature', type=list, default=[0], nargs='+', help='temperature.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--beta', type=float, default=0.1, help='the weight of kl')
parser.add_argument('--checkpoint', type=str, default=check_point, nargs='*',
                    help='initialize from a checkpoint, if None, do not restore')
parser.add_argument('--world_size', type=int, default=1, help='number of processes')
# args = parser.parse_args()
args = parser.parse_args(args=[])


def clear_atom_map_smiles(smiles_with_atom_map):
    mol = Chem.MolFromSmiles(smiles_with_atom_map)
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')
    return Chem.MolToSmiles(mol)


def need_atom_mapping(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return True

    for atom in mol.GetAtoms():
        atom_map_num = atom.GetAtomMapNum()
        if atom_map_num != 0:
            return False
    return True


class NERF_non_reactant_mask(BaseTool):
    name = 'NERF_non_reactant_mask'
    description = (
                'Use NERF_non_reactant_mask tool to Predict the product of a chemical reaction when only reactants are known and the reagents are unknown. ' +
                ' input SMILES string only, output the change of bonds and the SMILES of predicted products')
    checkpoint = "epoch-98-loss-1.1171849480149887"

    def map_atoms_in_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        atom_map = {}
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
            atom_map[atom.GetIdx()] = atom.GetIdx() + 1

        mapped_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return mapped_smiles

    def process_smiles(self, smiles, reactant_mask=None):
        # 需要原子映射处理, map_atoms=True
        # reactant_mask 是长度为“分子个数”的list, 1 for reactant, 0 for reagent
        map_atoms = need_atom_mapping(smiles)
        if map_atoms:
            smiles_atom_mapped = self.map_atoms_in_smiles(smiles)
            smiles_no_atom_mapped = smiles
        else:
            smiles_no_atom_mapped = clear_atom_map_smiles(smiles)
            smiles_atom_mapped = smiles

        reactant_mols = [Chem.MolFromSmiles(item) for item in smiles_atom_mapped.split(".")]
        reactant_len = Chem.MolFromSmiles(smiles_atom_mapped).GetNumAtoms()

        reactant_features = molecule(reactant_mols, reactant_len, reactant_mask)

        element = reactant_features['element']
        mask = reactant_features['mask']
        bond = reactant_features['bond']
        aroma = reactant_features['aroma']
        charge = reactant_features['charge']

        input_data = {}
        for key in reactant_features:
            if key in ["element", "reactant"]:
                input_data[key] = reactant_features[key]
            else:
                input_data['src_' + key] = reactant_features[key]

        data = [input_data]

        full_dataset = TransformerDataset(False, data)

        data_loader = DataLoader(full_dataset,
                                 batch_size=1,
                                 num_workers=4, collate_fn=TransformerDataset.collate_fn)

        return data_loader, element, mask, bond, aroma, charge, smiles_atom_mapped, smiles_no_atom_mapped

    def init_model(self, save_path, checkpoint):
        state_dict = {}
        map_location = {'cuda:%d' % 0: 'cuda:%d' % 0}
        checkpoint = torch.load(os.path.join(save_path, checkpoint), map_location=map_location)
        for key in checkpoint['model_state_dict']:
            if key in state_dict:
                state_dict[key] += checkpoint['model_state_dict'][key]
            else:
                state_dict[key] = checkpoint['model_state_dict'][key]

        model = MoleculeVAE(args, 100, 192, 6).to('cpu')  # TODO
        model.load_state_dict(state_dict)

        return model

    def predict(self, data_loader,
                save_path='../chemagent/tools/NERF/CKPT/no_reactant_mask/tmp/', checkpoint=checkpoint, temperature=0):

        model = self.init_model(save_path, checkpoint)

        for data in data_loader:  # 只有1个
            data_gpu = {}
            for key in data:
                data_gpu[key] = data[key].to('cpu')

            predicted_dict = model('sample', data_gpu, temperature)

            element = data['element']
            src_mask = data['src_mask']
            pred_bond = predicted_dict['bond'].cpu()
            pred_aroma, pred_charge = predicted_dict['aroma'].cpu(), predicted_dict['charge'].cpu()

            arg_list = [(element[j], src_mask[j], pred_bond[j], pred_aroma[j], pred_charge[j], None) for j in
                        range(1)]

            res = map(result2mol, arg_list)
            res = list(res)

            for item in res:
                mol, smile, valid, smile_no_map = item[0], item[1], item[2], item[3]

            return mol, smile, valid, pred_bond, pred_aroma, pred_charge, smile_no_map

    def pred_from_smiles(self, smiles,
                         reactant_mask=None):  # TODO: return mol_ori(atom_mapping & same idx with element) and mol_pred
        # processed_smile是一个经过了原子映射的smile
        dl, element, src_mask, src_bond, src_aroma, src_charge, smile_atom_map, smile_no_atom_map = self.process_smiles(
            smiles, reactant_mask)
        arg_list_src = [(element, src_mask, src_bond, src_aroma, src_charge, None)]
        result_src = map(result2mol, arg_list_src)
        result_src = list(result_src)
        mol_ori = None
        for item in result_src:
            mol_ori = item[0]
            src_mol_adj = Chem.GetAdjacencyMatrix(mol_ori)

        mol_pred, pred_smile_atom_mapping, pred_valid, pred_bond, pred_aroma, pred_charge, pred_smile_no_atom_map = self.predict(
            dl)

        pred_mol_adj = Chem.GetAdjacencyMatrix(mol_pred)
        diff_adj = pred_mol_adj - src_mol_adj

        return smile_atom_map, smile_no_atom_map, pred_smile_atom_mapping, pred_smile_no_atom_map, element, diff_adj, \
            mol_ori, mol_pred

    def get_reaction_info(self, elements, diff):
        element_symbols = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
                           "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Ni", "Co", "Cu", "Zn", "Ga", "Ge", "As",
                           "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
                           "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu",
                           "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
                           "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np",
                           "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs",
                           "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]

        reaction_info = []

        for i in range(len(diff)):
            for j in range(i + 1, len(diff)):
                if diff[i][j] >= 1:
                    reaction_info.append(
                        f"Formation of bond between atom {i + 1} ({element_symbols[elements[i] - 1]} element) and atom {j + 1} ({element_symbols[elements[j] - 1]} element)")
                elif diff[i][j] <= -1:
                    reaction_info.append(
                        f"Breaking of bond between atom {i + 1} ({element_symbols[elements[i] - 1]} element) and atom {j + 1} ({element_symbols[elements[j] - 1]} element)")

        return reaction_info

    def draw_reaction_graph(self, mol_ori, mol_pred, diff, pic_name_ori):

        # for mol_ori: highlight the bonds that are broken
        for i in range(len(diff)):
            for j in range(i + 1, len(diff)):
                if diff[i][j] == -1:
                    mol_ori.GetBondBetweenAtoms(i, j).SetProp("bondNote", "Broken")

        d2d = rdMolDraw2D.MolDraw2DSVG(350, 300)
        d2d.drawOptions().addAtomIndices = True
        d2d.drawOptions().setHighlightColour((0.8, 0.8, 0.8))
        d2d.DrawMolecule(mol_ori, highlightAtoms=[], highlightBonds=[0, 1])
        d2d.FinishDrawing()
        svg = d2d.GetDrawingText()
        with open('../tests/image/' + pic_name_ori + '.svg', 'w') as f:
            f.write(svg)

        # for mol_pred: highlight the bonds that are formed
        for i in range(len(diff)):
            for j in range(i + 1, len(diff)):
                if diff[i][j] == 1:
                    mol_pred.GetBondBetweenAtoms(i, j).SetProp("bondNote", "Formed")

        d2d = rdMolDraw2D.MolDraw2DSVG(350, 300)
        d2d.drawOptions().addAtomIndices = True
        d2d.drawOptions().setHighlightColour((0.8, 0.8, 0.8))
        d2d.DrawMolecule(mol_pred, highlightAtoms=[], highlightBonds=[0, 1])
        d2d.FinishDrawing()
        # SVG(d2d.GetDrawingText())
        svg = d2d.GetDrawingText()
        with open('../tests/image/' + 'predicted products of' + pic_name_ori + '.svg', 'w') as f:
            f.write(svg)

    def _run(self, reactants_smile: str) -> str:
        # smile_atom_map, smile_no_atom_map, pred_smile_atom_mapping, pred_smile_no_atom_map, element, diff_adj
        reactant_mask = None
        smile_atom_map, smile_no_atom_map, pred_smile_atom_mapping, pred_smile_no_atom_mapping, src_element, diff_, \
            mol_ori, mol_pred = self.pred_from_smiles(reactants_smile, reactant_mask=reactant_mask)
        explain = self.get_reaction_info(src_element, diff_)
        self.draw_reaction_graph(mol_ori, mol_pred, diff_, smile_no_atom_map)
        return '\n'.join(['reactants SMILES:' + smile_no_atom_map,
                          'reactants after atom mapping:' + smile_atom_map,
                          'predicted products with atom mapping:' + pred_smile_atom_mapping,
                          'predicted products without atom mapping:' + pred_smile_no_atom_mapping,
                          'Changes in Covalent Bonds in Reactions'
                          ] + explain)


class ReactionT5(BaseTool):
    name = 'reaction_t5'
    description = ('Predict the product of a chemical reaction')

    def __init__(self, model_name="t5-large", verbose=False):
        super().__init__(verbose=verbose)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained('sagawa/ReactionT5-product-prediction')
        self.model = T5ForConditionalGeneration.from_pretrained('sagawa/ReactionT5-product-prediction')

    def _run(self, input_smiles: str) -> str:
        inp = self.tokenizer(f'REACTANT:{input_smiles}REAGENT:', return_tensors='pt')
        output = self.model.generate(**inp, min_length=6, max_length=109, num_beams=1, num_return_sequences=1,
                                     return_dict_in_generate=True, output_scores=True)
        output = self.tokenizer.decode(output['sequences'][0], skip_special_tokens=True).replace(' ', '').rstrip('.')
        return output

    async def _arun(self, smiles_pair: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()


class NERF_know_reagents(BaseTool):
    name = 'NERF_know_reagents'
    description = (
                'Given the reactants and reagents (where reagents can be None) of a chemical reaction, Predict the product of a chemical ' +
                'reaction. Input SMILES string of reactants and reagents separately, output the change of bonds ' +
                'and the SMILES of predicted products.' +
                'Strictly follow the input format, this is an example of input:"Reactants:CS(=O)(=O)OC[C@H]1CCC(=O)O1 Reagents:Fc1ccc(Nc2ncnc3cc(OCCN4CCNCC4)c(OC4CCCC4)cc23)cc1Cl"' +
                'You should replace the SMILES string with your own targeted SMILES string that given in the question.')
    # checkpoint = "epoch-232-loss-0.5991340563215058"
    # checkpoint = "epoch-171-loss-0.6919782893504469"
    checkpoint = "epoch-199-loss-0.5907546520840412"
    def map_atoms_in_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        atom_map = {}
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
            atom_map[atom.GetIdx()] = atom.GetIdx() + 1

        mapped_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return mapped_smiles

    def process_smiles(self, smiles, reactant_mask):
        # 需要原子映射处理, map_atoms=True
        # reactant_mask 是长度为“分子个数”的list, 1 for reactant, 0 for reagent
        map_atoms = need_atom_mapping(smiles)
        if map_atoms:
            smiles_atom_mapped = self.map_atoms_in_smiles(smiles)
            smiles_no_atom_mapped = smiles
        else:
            smiles_no_atom_mapped = clear_atom_map_smiles(smiles)
            smiles_atom_mapped = smiles

        reactant_mols = [Chem.MolFromSmiles(item) for item in smiles_atom_mapped.split(".")]
        reactant_len = Chem.MolFromSmiles(smiles_atom_mapped).GetNumAtoms()

        reactant_features = molecule(reactant_mols, reactant_len, reactant_mask)

        element = reactant_features['element']
        mask = reactant_features['mask']
        bond = reactant_features['bond']
        aroma = reactant_features['aroma']
        charge = reactant_features['charge']

        input_data = {}
        for key in reactant_features:
            if key in ["element", "reactant"]:
                input_data[key] = reactant_features[key]
            else:
                input_data['src_' + key] = reactant_features[key]

        data = [input_data]

        full_dataset = TransformerDataset(False, data)

        data_loader = DataLoader(full_dataset,
                                 batch_size=1,
                                 num_workers=4, collate_fn=TransformerDataset.collate_fn)

        return data_loader, element, mask, bond, aroma, charge, smiles_atom_mapped, smiles_no_atom_mapped

    def init_model(self, save_path, checkpoint):
        state_dict = {}
        map_location = {'cuda:%d' % 0: 'cuda:%d' % 0}
        checkpoint = torch.load(os.path.join(save_path, checkpoint), map_location=map_location)
        for key in checkpoint['model_state_dict']:
            if key in state_dict:
                state_dict[key] += checkpoint['model_state_dict'][key]
            else:
                state_dict[key] = checkpoint['model_state_dict'][key]

        model = MoleculeVAE(args, 100, 192, 6).to('cpu')  # TODO
        model.load_state_dict(state_dict)

        return model

    def predict(self, data_loader,
                save_path='../chemagent/tools/NERF/CKPT/', checkpoint=checkpoint, temperature=0):

        model = self.init_model(save_path, checkpoint)

        for data in data_loader:  # 只有1个
            data_gpu = {}
            for key in data:
                data_gpu[key] = data[key].to('cpu')

            predicted_dict = model('sample', data_gpu, temperature)

            element = data['element']
            src_mask = data['src_mask']
            pred_bond = predicted_dict['bond'].cpu()
            pred_aroma, pred_charge = predicted_dict['aroma'].cpu(), predicted_dict['charge'].cpu()

            arg_list = [(element[j], src_mask[j], pred_bond[j], pred_aroma[j], pred_charge[j], None) for j in
                        range(1)]

            res = map(result2mol, arg_list)
            res = list(res)

            for item in res:
                mol, smile, valid, smile_no_map, adj_matrix_pred = item[0], item[1], item[2], item[3], item[4]

            return mol, smile, valid, pred_bond, pred_aroma, pred_charge, smile_no_map, adj_matrix_pred

    def pred_from_reactants_reagents(self, reactants, reagents):
        # processed_smile是一个经过了原子映射的smile
        smiles = '.'.join([reactants, reagents])
        if reagents is not None:
            reactant_mask = [1 for _ in reactants.split('.')] + [0 for _ in reagents.split('.')]
        else:
            reactant_mask = [1 for _ in reactants.split('.')]
        dl, element, src_mask, src_bond, src_aroma, src_charge, smile_atom_map, smile_no_atom_map = self.process_smiles(
            smiles, reactant_mask)
        arg_list_src = [(element, src_mask, src_bond, src_aroma, src_charge, None)]
        result_src = map(result2mol, arg_list_src)
        result_src = list(result_src)
        for item in result_src:
            mol_ori = item[0]
            adj_matrix_ori = item[4]
            src_mol_adj = Chem.GetAdjacencyMatrix(item[0]) # TODO check this; result2mol 返回 adj_matrix_ori,  predict函数返回adj_matrix_pred

        pred_mol, pred_smile_atom_mapping, pred_valid, pred_bond, pred_aroma, pred_charge, pred_smile_no_atom_map, adj_matrix_pred = self.predict(
            dl)

        pred_mol_adj = Chem.GetAdjacencyMatrix(pred_mol)
        diff_adj = adj_matrix_pred.adj_matrix.numpy() - adj_matrix_ori.adj_matrix.numpy()

        return smile_atom_map, smile_no_atom_map, pred_smile_atom_mapping, pred_smile_no_atom_map, element, diff_adj, \
            mol_ori, pred_mol

    def get_reaction_info(self, elements, diff):
        element_symbols = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
                           "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Ni", "Co", "Cu", "Zn", "Ga", "Ge", "As",
                           "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
                           "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu",
                           "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
                           "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np",
                           "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs",
                           "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]

        reaction_info = []

        for i in range(len(diff)):
            for j in range(i + 1, len(diff)):
                if diff[i][j] == 1:
                    reaction_info.append(
                        f"Formation of bond between atom {i + 1} ({element_symbols[elements[i] - 1]} element) and atom {j + 1} ({element_symbols[elements[j] - 1]} element)")
                elif diff[i][j] == -1:
                    reaction_info.append(
                        f"Breaking of bond between atom {i + 1} ({element_symbols[elements[i] - 1]} element) and atom {j + 1} ({element_symbols[elements[j] - 1]} element)")

        return reaction_info

    def draw_reaction_graph1(self, mol_ori, mol_pred, diff, pic_name_ori):
        # for mol_ori: highlight the bonds that are broken
        bond_hilights_ori = []
        for i in range(len(diff)):
            for j in range(i + 1, len(diff)):
                if diff[i][j] == -1:
                    mol_ori.GetBondBetweenAtoms(i, j).SetProp("bondNote", "Broken")
                    bond_hilights_ori.append(mol_ori.GetBondBetweenAtoms(i, j).GetIdx())

        # for mol_pred: highlight the bonds that are formed
        bond_hilights_pred = []
        for i in range(len(diff)):
            for j in range(i + 1, len(diff)):
                if diff[i][j] == 1:
                    mol_pred.GetBondBetweenAtoms(i, j).SetProp("bondNote", "Formed")
                    bond_hilights_pred.append(mol_pred.GetBondBetweenAtoms(i, j).GetIdx())

        # Create an RDKit Mol object for the second molecule
        mol_pred_highlighted = Chem.Mol(mol_pred)

        # Highlight bonds in the second molecule
        for bond_idx in bond_hilights_pred:
            mol_pred_highlighted.GetBondWithIdx(bond_idx).SetProp("bondNote", "Formed")

        # Create an RDKit Mol object for the first molecule
        mol_ori_highlighted = Chem.Mol(mol_ori)

        # Highlight bonds in the first molecule
        for bond_idx in bond_hilights_ori:
            mol_ori_highlighted.GetBondWithIdx(bond_idx).SetProp("bondNote", "Broken")

        # Create a grid image with both molecules and their highlights
        img = Draw.MolsToGridImage([mol_ori_highlighted, mol_pred_highlighted],
                                   molsPerRow=2, subImgSize=(200, 200),
                                   legends=['Original', 'Predicted'],
                                   useSVG=True)
        with open('../tests/image/' + '1reaction:' + pic_name_ori + '.svg', 'w') as f:
            f.write(img.data)
        # img.save('../tests/image/' + 'reaction_comparison_highlighted_' + pic_name_ori + '.png')

    def draw_reaction_graph(self, mol_ori, mol_pred, reagents_smiles, diff, pic_name_ori):
        if len(pic_name_ori) > 20:
            pic_name_ori = pic_name_ori[:20]

        # reagents_mol = Chem.MolFromSmiles(reagents_smiles)
        ori_smiles = Chem.MolToSmiles(mol_ori).split('.')
        pred_smiles = Chem.MolToSmiles(mol_pred).split('.')
        reagents_smiles = []
        for o in ori_smiles:
            if o in pred_smiles:
                reagents_smiles.append(o)

        reagents_smiles = Chem.MolFromSmiles('.'.join(reagents_smiles))

        d2d = rdMolDraw2D.MolDraw2DSVG(350, 300)
        d2d.drawOptions().setHighlightColour((1.0, 0.5, 0.0))
        d2d.DrawMolecule(reagents_smiles)
        d2d.FinishDrawing()
        svg = d2d.GetDrawingText()
        with open('../tests/image/reagents_' + pic_name_ori + '.svg', 'w') as f:
            f.write(svg)

        # for mol_ori: highlight the bonds that are broken
        bond_hilights = []
        for i in range(len(diff)):
            for j in range(i + 1, len(diff)):
                if diff[i][j] <= -1:
                    mol_ori.GetBondBetweenAtoms(i, j).SetProp("bondNote", "Broken")
                    bond_hilights.append(mol_ori.GetBondBetweenAtoms(i, j).GetIdx())

        atom_hilights_ori = [i for i in range(len(mol_ori.GetAtoms())) if
                             any(diff[i][j] <= -1 for j in range(len(diff)))]

        mol_ori = Chem.DeleteSubstructs(mol_ori, reagents_smiles, onlyFrags=True)

        d2d = rdMolDraw2D.MolDraw2DSVG(350, 300)
        d2d.drawOptions().addAtomIndices = True
        d2d.drawOptions().setHighlightColour((1.0, 0.5, 0.0))
        d2d.DrawMolecule(mol_ori, highlightAtoms=atom_hilights_ori, highlightBonds=bond_hilights)
        d2d.FinishDrawing()
        svg = d2d.GetDrawingText()
        with open('../tests/image/' + pic_name_ori + '.svg', 'w') as f:
            f.write(svg)

        # for mol_pred: highlight the bonds that are formed
        bond_hilights = []
        for i in range(len(diff)):
            for j in range(i + 1, len(diff)):
                if diff[i][j] >= 1:
                    mol_pred.GetBondBetweenAtoms(i, j).SetProp("bondNote", "Formed")
                    bond_hilights.append(mol_pred.GetBondBetweenAtoms(i, j).GetIdx())

        atom_hilights_pred = [i for i in range(len(mol_ori.GetAtoms())) if
                             any(diff[i][j] >= 1 for j in range(len(diff)))]

        mol_pred = Chem.DeleteSubstructs(mol_pred, reagents_smiles, onlyFrags=True)

        d2d = rdMolDraw2D.MolDraw2DSVG(350, 300)
        d2d.drawOptions().addAtomIndices = True
        d2d.drawOptions().setHighlightColour((1.0, 0.5, 0.0))
        d2d.DrawMolecule(mol_pred, highlightAtoms=atom_hilights_pred, highlightBonds=bond_hilights)
        d2d.FinishDrawing()
        # SVG(d2d.GetDrawingText())
        svg = d2d.GetDrawingText()
        with open('../tests/image/' + 'predicted products of' + pic_name_ori + '.svg', 'w') as f:
            f.write(svg)

    def draw_reaction(self, mol_ori, mol_pred, pic_name_ori):
        if len(pic_name_ori) > 20:
            pic_name_ori = pic_name_ori[:20]

        smile_ori = Chem.MolToSmiles(mol_ori)
        smile_pred = Chem.MolToSmiles(mol_pred)

        rxn = AllChem.ReactionFromSmarts(smile_ori + '>>' + smile_pred)
        d2d = Draw.MolDraw2DCairo(800, 400)
        d2d.DrawReaction(rxn, highlightByReactant=True)

        # svg = d2d.GetDrawingText()
        # with open('../tests/image/' + 'reaction_' + pic_name_ori + '.svg', 'w') as f:
        #     f.write(svg)

        import io
        bio = io.BytesIO(d2d.GetDrawingText())
        open('../tests/image/' + 'reaction_' + pic_name_ori + '.png', 'wb+').write(bio.getvalue())

    def _run(self, input_string: str) -> str:
        # smile_atom_map, smile_no_atom_map, pred_smile_atom_mapping, pred_smile_no_atom_map, element, diff_adj
        cleaned_input = input_string.replace(" ", "").lower()
        cleaned_input1 = input_string.replace(" ", "")

        reactants_index = cleaned_input.find("reactants:")
        reagents_index = cleaned_input.find("reagents:")

        if reactants_index == -1 or reagents_index == -1:
            return None, None

        reactants = cleaned_input1[reactants_index + 10:reagents_index].strip()
        reagents = cleaned_input1[reagents_index + 9:].strip()

        smile_atom_map, smile_no_atom_map, pred_smile_atom_mapping, pred_smile_no_atom_mapping, src_element, diff_, \
            mol_ori, mol_pred = self.pred_from_reactants_reagents(reactants, reagents)
        self.draw_reaction_graph(mol_ori, mol_pred, reagents, diff_, smile_no_atom_map)
        self.draw_reaction(mol_ori, mol_pred, smile_no_atom_map)
        explain = self.get_reaction_info(src_element, diff_)
        return '\n'.join(['reactants SMILES:' + smile_no_atom_map,
                          'reactants after atom mapping:' + smile_atom_map,
                          'predicted products with atom mapping:' + pred_smile_atom_mapping,
                          'predicted products without atom mapping:' + pred_smile_no_atom_mapping,
                          'Changes in Covalent Bonds in Reactions'
                          ] + explain)
