"""Core tests."""
import unittest

import yaml
from oaklib import get_implementation_from_shorthand

from semantic_llama.clients.pubmed_client import PubmedClient
from semantic_llama.engines import create_engine
from semantic_llama.engines.text_model_knowledge_engine import TextModelKnowledgeEngine
from semantic_llama.templates.biological_process import BiologicalProcess
from semantic_llama.templates.gocam import ExtractionResult, GoCamAnnotations

TEMPLATE = "gocam.GoCamAnnotations"

PAPER = """
Title: β-Catenin Is Required for the cGAS/STING Signaling Pathway but Antagonized by the Herpes Simplex Virus 1 US3 Protein
Text:
The cGAS/STING-mediated DNA-sensing signaling pathway is crucial
for interferon (IFN) production and host antiviral
responses. Herpes simplex virus I (HSV-1) is a DNA virus that has
evolved multiple strategies to evade host immune responses. Here,
we demonstrate that the highly conserved β-catenin protein in the
Wnt signaling pathway is an important factor to enhance the
transcription of type I interferon (IFN-I) in the cGAS/STING
signaling pathway, and the production of IFN-I mediated by
β-catenin was antagonized by HSV-1 US3 protein via its kinase
activity. Infection by US3-deficienct HSV-1 and its kinase-dead
variants failed to downregulate IFN-I and IFN-stimulated
gene (ISG) production induced by β-catenin. Consistent with this,
absence of β-catenin enhanced the replication of US3-deficienct
HSV-1, but not wild-type HSV-1. The underlying mechanism was the
interaction of US3 with β-catenin and its hyperphosphorylation of
β-catenin at Thr556 to block its nuclear translocation. For the
first time, HSV-1 US3 has been shown to inhibit IFN-I production
through hyperphosphorylation of β-catenin and to subvert host
antiviral innate immunity.IMPORTANCE Although increasing evidence
has demonstrated that HSV-1 subverts host immune responses and
establishes lifelong latent infection, the molecular mechanisms
by which HSV-1 interrupts antiviral innate immunity, especially
the cGAS/STING-mediated cellular DNA-sensing signaling pathway,
have not been fully explored. Here, we show that β-catenin
promotes cGAS/STING-mediated activation of the IFN pathway, which
is important for cellular innate immune responses and intrinsic
resistance to DNA virus infection. The protein kinase US3
antagonizes the production of IFN by targeting β-catenin via its
kinase activity. The findings in this study reveal a novel
mechanism for HSV-1 to evade host antiviral immunity and add new
knowledge to help in understanding the interaction between the
host and HSV-1 infection.

Keywords: HSV-1; US3; type I IFN; β-catenin.
"""

EXAMPLE_RESULTS = """
genes: β-Catenin; cGAS; STING; US3; IFN; ISG
organisms: Herpes Simplex Virus I (HSV-1);
gene_organisms: β-Catenin:host; cGAS:host; STING:host; US3:HSV-1; IFN:host; ISG:host
activities: production of type I IFN; transcription of type I IFN; replication of HSV-1; nuclear translocation of β-catenin
gene_functions: β-catenin:enhance the transcription of type I IFN; US3:antagonize the production of IFN; β-catenin:block nuclear translocation
cellular_processes: cGAS/STING-mediated DNA-sensing signaling; activation of IFN pathway
pathways: IFN pathway; Wnt signalling pathway
gene_gene_interactions: US3:β-catenin
gene_localizations: US3:host; β-catenin:host
"""

EXAMPLE_RESULTS_ALT = """
genes: β-Catenin; cGAS; STING; US3; IFN; ISG
organisms: Herpes Simplex Virus I (HSV-1);
gene_organisms: β-Catenin - Human; cGAS - Human; STING - Human; US3 - Human; IFN - Human; ISG - Human
activities: Transcription; Production; Downregulation; Replication; Nuclear Translocation
gene_functions: β-Catenin - Enhances Transcription; US3 - Antagonizes Production; US3 - Downregulates IFN-I; US3 - Blocks Nuclear Translocation; β-Catenin - Enhances Production
cellular_processes: DNA-sensing; Interferon Production; Antiviral Innate Immunity; Host Innate Immune Responses; Interaction with Host; Evade Host Antiviral Immunity
pathways: cGAS/STING-mediated DNA-sensing; Wnt Signaling; IFN pathway
gene_gene_interactions: US3 - β-Catenin; β-Catenin - US3
gene_localizations: β-Catenin - Nuclear; US3 - Hyperphosphorylation
"""

TEST_PROCESS = BiologicalProcess(
    label="autophagosome assembly",
    description="The formation of a double membrane-bounded structure, the autophagosome, that occurs when a specialized membrane sac, called the isolation membrane, starts to enclose a portion of the cytoplasm",
    subclass_of="GO:0022607",
    outputs=["GO:0005776"],
)

RAW_PARSE = {
    "genes": ["β-Catenin", "cGAS", "STING", "US3", "IFN", "ISG"],
    "gene_organisms": [
        ("β-Catenin", "host"),
        ("cGAS", "host"),
        ("STING", "host"),
        ("US3", "HSV-1"),
        ("IFN", "host"),
        ("ISG", "host"),
    ],
}


class TestCore(unittest.TestCase):
    """Test annotation."""

    def setUp(self) -> None:
        """Set up."""
        self.ke = create_engine(TEMPLATE, TextModelKnowledgeEngine)

    def test_setup(self):
        """Tests template and module is loaded"""
        ke = self.ke
        pyc = ke.template_pyclass
        print(pyc)
        obj = pyc(genes=["a"], gene_organisms=[{"gene": "a", "organism": "b"}])
        print(yaml.dump(obj.dict()))
        self.assertEqual(obj.genes, ["a"])
        self.assertEqual(obj.gene_organisms[0].gene, "a")
        self.assertEqual(obj.gene_organisms[0].organism, "b")
        slot = ke.schemaview.induced_slot("genes", "GeneOrganismRelationship")
        self.assertEqual(slot.name, "genes")
        self.assertEqual(slot.multivalued, True)
        self.assertEqual(slot.range, "Gene")

    def test_generalize(self):
        """Tests generalization."""
        ke = TextModelKnowledgeEngine("biological_process.BiologicalProcess")
        ke.labelers = [get_implementation_from_shorthand("sqlite:obo:go")]
        examples = [
            """
            label: mannose biosynthesis
            description: The chemical reactions and pathways resulting in the formation of mannose, the aldohexose manno-hexose, the C-2 epimer of glucose
            synonyms: mannose anabolism
            subclass_of: hexose biosynthesis
            outputs: mannose
            """,
            """
            label: maltose biosynthesis
            description: The chemical reactions and pathways resulting in the formation of the disaccharide maltose (4-O-alpha-D-glucopyranosyl-D-glucopyranose)
            subclass_of: disaccharide biosynthesis
            outputs: maltose
            """,
            TEST_PROCESS,
        ]
        label = "beta-D-lyxofuranose biosynthesis"
        ann = ke.generalize({"label": label}, examples)
        print(f"RESULTS={ann}")
        print(yaml.dump(ann.dict()))
        self.assertEqual(label, ann.label)
        self.assertEqual(["CHEBI:151400"], ann.outputs)

    def test_serialize_example(self):
        ke = TextModelKnowledgeEngine("biological_process.BiologicalProcess")
        ke.labelers = [get_implementation_from_shorthand("sqlite:obo:go")]
        ser = ke.serialize_object(TEST_PROCESS)
        # print(f"SERIALIZED={ser}")
        self.assertIn("outputs: autophagosome", ser)

    def test_extract(self):
        """Tests end to end knowledge extraction."""
        ke = self.ke
        ann = ke.extract_from_text(PAPER)
        print(f"RESULTS={ann}")
        print(yaml.dump(ann.dict()))
        results = ann.results
        if not isinstance(results, GoCamAnnotations):
            raise ValueError(f"Expected GoCamAnnotations, got {type(results)}")
        self.assertIn("HGNC:2514", results.genes)

    def test_extract_with_stub(self):
        """Tests end to end knowledge extraction."""
        ke = self.ke
        ann = ke.extract_from_text(PAPER, object={"pathways": ["GO:0140896"]})
        print(f"RESULTS={ann}")
        print(yaml.dump(ann.dict()))
        results = ann.results
        if not isinstance(results, GoCamAnnotations):
            raise ValueError(f"Expected GoCamAnnotations, got {type(results)}")
        self.assertIn("HGNC:2514", results.genes)

    def test_subextract(self):
        """Tests end to end knowledge extraction."""
        ke = self.ke
        cls = ke.schemaview.get_class("GeneMolecularActivityRelationship")
        ann = ke.extract_from_text("β-Catenin-Translocation", cls)
        print(f"RESULTS={ann}")
        print(yaml.dump(ann.dict()))
        self.assertEqual({"gene": "HGNC:2514", "molecular_activity": "Translocation"}, ann.dict())
        # try and fool it
        ann = ke._extract_from_text_to_dict("foobaz", cls)
        print(f"RESULTS={ann}")
        self.assertIsNone(ann)
        # print(yaml.dump(ann.dict()))

    def test_prompt(self):
        """Tests prompt generation.

        Tests that the prompt can be derived from a prompt annotation,
        or from description, if no such annotation present
        """
        ke = self.ke
        prompt = ke.get_completion_prompt()
        self.assertIn(
            "genes: <semicolon-separated list of genes>",
            prompt,
            "Expected to derived prompt from description of gene slot",
        )
        self.assertIn(
            "gene_organisms: <semicolon-separated list of asterisk separated gene to organism relationships>",
            prompt,
            "Expected to derived prompt from annotations of gene_organisms slot",
        )

    def test_parse_completion_payload(self):
        """Tests parsing of textual payload from openai API.

        This involves two steps:

        - parsing the payload into a dict of slot values
        - annotating the dict using OAK annotators

        We use a ready-made payload from the API, to avoid
        invoking an API call
        """
        ke = self.ke
        ann = ke.parse_completion_payload(EXAMPLE_RESULTS)
        print(f"PARSED={ann}")
        print(yaml.dump(ann.dict()))
        self.assertIn("HGNC:2514", ann.genes)
        self.assertEqual(2, len(ann.pathways))

    def test_parse_response_to_dict(self):
        """Tests parsing of textual payload from openai API."""
        ke = self.ke
        ann = ke._parse_response_to_dict(EXAMPLE_RESULTS)
        print(f"PARSED={ann}")
        print(yaml.dump(ann))
        self.assertIn("STING", ann["genes"])
        self.assertIn({"gene": "β-Catenin", "organism": "host"}, ann["gene_organisms"])
        # test resilience to missing internal separators
        ann = ke._parse_response_to_dict("gene_organisms: a ; b ; c\ngenes: g")
        self.assertEqual(ann["genes"], ["g"])
        self.assertEqual(["genes"], list(ann.keys()))
        # test resilience to multiple internal separators
        ann = ke._parse_response_to_dict("gene_organisms: a-b-c")
        self.assertEqual(ann["gene_organisms"], [{"gene": "a", "organism": "b-c"}])

    def test_parse2(self):
        """Tests parsing of textual payload from openai API.

        This uses a variant of the output, sometimes OpenAI will
        use ':' as separators instead of ' - '.
        """
        ke = self.ke
        ann = ke.parse_completion_payload(EXAMPLE_RESULTS_ALT)
        print(f"PARSED={ann}")
        print(yaml.dump(ann))

    def test_ground_annotation_object(self):
        """
        Tests grounding of annotation object.

        E.g. takes

            {'genes': ['β-Catenin', 'cGAS', ...]}

        And produces:

            {'genes': ['HGNC:2514', 'HGNC:21367', ...
        """
        ke = self.ke
        annotated = ke.ground_annotation_object(RAW_PARSE)
        print(yaml.dump(annotated.dict()))

    def test_mapping(self):
        """Tests mapping of annotation object to OAK model."""
        ke = self.ke
        cases = [
            (
                "uberon",
                [
                    "pinky",
                    "brain",
                    "cerebellum",
                    "feet",
                    "epithelia",
                    "skin",
                    "hair",
                    "lungs",
                    "pancreatic",
                    "nerves",
                    "nostrils",
                ],
            ),
            (
                "go",
                [
                    "Wnt signaling",
                    "apoptosis",
                    "lactase",
                    "positive regulation of alpha-glucoside transport",
                    "development of lungs",
                ],
            ),
        ]
        for ontology, terms in cases:
            ann = ke.map_terms(terms, ontology)
            print(yaml.dump(ann))

    def test_extract_from_pubmed(self):
        """Tests end to end knowledge extraction."""
        ke = self.ke
        pmc = PubmedClient()
        text = pmc.text("PMID:27929086")
        print(text)
        ann = ke.extract_from_text(PAPER)
        print(f"RESULTS={ann}")
        print(yaml.dump(ann.dict()))

    def test_merge(self):
        bp1 = BiologicalProcess(
            label="bp",
            description="d1",
            synonyms=["s1a", "s1b", "shared"],
            inputs=["c1", "c2"],
        )
        bp2 = BiologicalProcess(
            label="bp",
            description="d2",
            synonyms=["s2a", "s2b", "shared"],
            outputs=["c3", "c4"],
        )
        r1 = ExtractionResult(results=bp1)
        r2 = ExtractionResult(results=bp2)
        bp = self.ke.merge_resultsets([r1, r2]).results
        print(yaml.dump(bp.dict()))
        self.assertEqual("bp", bp.label)
        self.assertEqual("d2", bp.description)
        self.assertCountEqual(["s1a", "s1b", "s2a", "s2b", "shared"], bp.synonyms)
        self.assertEqual(5, len(bp.synonyms), "Expected 5 distinct synonyms")
        self.assertCountEqual(["c1", "c2", "c3", "c4"], bp.inputs + bp.outputs)
        self.assertCountEqual(["c1", "c2"], bp.inputs)
        self.assertCountEqual(["c3", "c4"], bp.outputs)

    def test_merge_with_pmids(self):
        """Tests end to end knowledge extraction."""
        ke = self.ke
        pmids = ["32716108", "27929086"]
        pmc = PubmedClient()
        rsets = []
        for pmid in pmids:
            text = pmc.text(f"PMID:{pmid}")
            rsets.append(ke.extract_from_text(text))
        print(f"Merging {len(rsets)} results")
        merged = ke.merge_resultsets(rsets)
        print(yaml.dump(merged.dict()))
