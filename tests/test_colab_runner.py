"""Offline tests for the self-contained notebook and its scientific adapter."""
import ast
import csv
import ctypes
import ctypes.util
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("esmfluc_notebook_runner", ROOT / "notebooks/esmfluc_runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def evaluate_javascript(source):
    """Run viewer logic with the system JS engine when available; no browser/network."""
    library = ctypes.util.find_library("javascriptcoregtk-4.0")
    if not library:
        raise unittest.SkipTest("JavaScriptCore is not available")
    lib = ctypes.CDLL(library)
    pointer = ctypes.c_void_p
    signatures = {
        "JSGlobalContextCreate": ([pointer], pointer),
        "JSGlobalContextRelease": ([pointer], None),
        "JSStringCreateWithUTF8CString": ([ctypes.c_char_p], pointer),
        "JSStringRelease": ([pointer], None),
        "JSEvaluateScript": ([pointer, pointer, pointer, pointer, ctypes.c_int, ctypes.POINTER(pointer)], pointer),
        "JSValueToStringCopy": ([pointer, pointer, ctypes.POINTER(pointer)], pointer),
        "JSStringGetMaximumUTF8CStringSize": ([pointer], ctypes.c_size_t),
        "JSStringGetUTF8CString": ([pointer, pointer, ctypes.c_size_t], ctypes.c_size_t),
    }
    for name, (args, result) in signatures.items():
        function = getattr(lib, name); function.argtypes = args; function.restype = result
    context = lib.JSGlobalContextCreate(None)
    script = lib.JSStringCreateWithUTF8CString(source.encode())
    exception = pointer()
    try:
        result = lib.JSEvaluateScript(context, script, None, None, 1, ctypes.byref(exception))
        text = lib.JSValueToStringCopy(context, exception.value or result, None)
        try:
            size = lib.JSStringGetMaximumUTF8CStringSize(text)
            buffer = ctypes.create_string_buffer(size)
            lib.JSStringGetUTF8CString(text, buffer, size)
            value = buffer.value.decode()
        finally:
            lib.JSStringRelease(text)
        if exception.value:
            raise AssertionError(value)
        return json.loads(value)
    finally:
        lib.JSStringRelease(script)
        lib.JSGlobalContextRelease(context)


class NotebookTests(unittest.TestCase):
    def test_notebook_is_clean_and_syntactically_valid(self):
        nb = json.loads((ROOT / "notebooks/ESMfluc.ipynb").read_text())
        self.assertEqual(nb["nbformat"], 4)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_embedded_runner_and_requirements_match_files(self):
        nb = json.loads((ROOT / "notebooks/ESMfluc.ipynb").read_text())
        for cell_id, name, filename in (
            ("adapter", "RUNNER_SOURCE", "esmfluc_runner.py"),
            ("setup", "REQUIREMENTS", "requirements-colab.txt"),
        ):
            cell = next(c for c in nb["cells"] if c["id"] == cell_id)
            tree = ast.parse("".join(cell["source"]))
            assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
            self.assertEqual(ast.literal_eval(assignment.value), (ROOT / "notebooks" / filename).read_text())

    def test_input_validation(self):
        self.assertEqual(runner.parse_sequence(">protein\nACD EFG\nHIX\n"), ("protein", "ACDEFGHIX"))
        for value in (">a\nACD\n>b\nEFG", "ACD123", "ACD*", "A" * 1025, "AA"):
            with self.assertRaises(ValueError):
                runner.parse_sequence(value)

    def test_ensemble_class_uses_mean_probability_not_mean_logit(self):
        margins = np.array([10.0, -2.0, -2.0])
        p = 1 / (1 + np.exp(-margins))
        probabilities = [np.array([[1-x, x]]) for x in p]
        mean, labels = runner.mean_predictions(probabilities)
        self.assertGreater(margins.mean(), 0)
        self.assertLess(mean[0, 1], .5)
        self.assertEqual(labels.tolist(), [0])

    def test_band_stability_requires_two_same_sign_matches(self):
        profile = np.array([0, .1, -.1, .2, 3, .2, -.1, -.2, .1, 0])
        rows = runner.detect_bands(profile, "ACDEFGHIKL", "test", "mean")
        self.assertEqual(len(rows), 1)
        self.assertEqual(runner.stable_bands(rows, {1: rows, 2: [], 3: []}), [])
        stable = runner.stable_bands(rows, {1: rows, 2: rows, 3: []})
        self.assertEqual(stable[0]["seed_support_count"], 2)
        self.assertEqual(stable[0]["supporting_seeds"], [1, 2])
        opposite = [dict(row, sign=-row["sign"]) for row in rows]
        self.assertEqual(runner.stable_bands(rows, {1: rows, 2: opposite, 3: []}), [])

    def test_flat_profile_has_no_bands(self):
        self.assertEqual(runner.detect_bands(np.zeros(10), "ACDEFGHIKL", "test", 1), [])

    def test_column_threshold_uses_magnitude_and_has_no_fallback_or_top_cap(self):
        function = runner.LINKED_VIEWER_JS.split("function receiversForKey", 1)[1].split("function baseStructure", 1)[0]
        result = evaluate_javascript("const data={sequence:'ACDE'}; function receiversForKey" + function + """
const matrix=[[0,.5,0,0],[0,-.8,0,0],[0,0,0,0],[0,.49,0,0]];
const signed=receiversForKey(matrix,2,.5);
const empty=receiversForKey(matrix,2,1);
const zero=receiversForKey(matrix,2,0);
const invalid=receiversForKey(matrix,2,-1);
data.sequence='A'.repeat(130);
const uncapped=receiversForKey(Array.from({length:130},()=>[1]),1,.5);
JSON.stringify({signed,empty,zero,invalid,n:uncapped.length});
""")
        self.assertEqual(result["signed"], [{"position": 2, "value": -.8}, {"position": 1, "value": .5}])
        self.assertEqual(result["empty"], [])
        self.assertEqual(len(result["zero"]), 3)
        self.assertEqual(result["invalid"], [])
        self.assertEqual(result["n"], 130)

    def test_viewer_events_link_sequence_and_mapped_structure(self):
        fixture = r"""
const events={},styles=[];
function element(){return {value:'',dataset:{},textContent:'',children:[],listeners:{},
  classList:{values:new Set(),toggle(c,on){if(on)this.values.add(c);else this.values.delete(c);},
    remove(...names){for(const n of names)this.values.delete(n);}},
  addEventListener(name,fn){this.listeners[name]=fn;},
  appendChild(child){this.children.push(child);},replaceChildren(){this.children=[];}};}
const nodes={};
for(const id of ['viewer-data','contribution-plot','threshold','source-position','selection-status',
  'receiver-rows','receiver-count','select-source','clear-selection','structure-note']) nodes[id]=element();
const buttons=Array.from({length:4},(_,i)=>{const el=element();el.dataset={pos:String(i+1),description:'residue'};return el;});
const mapping=Array.from({length:4},(_,i)=>({resolved:i!==2,pdb_chain:'A',pdb_residue:101+i,
  insertion_code:i===1?'A':'',structure_amino_acid:'ACDE'[i]}));
nodes['viewer-data'].textContent=JSON.stringify({sequence:'ACDE',has_contributions:true,
  structure:{pdb:'test',bands:[],mapping,atom_serials:{1:[101],2:[102],4:[104]},ca_serials:{1:101,2:102,4:104}}});
nodes.threshold.value='.5';
nodes['contribution-plot'].data=[{z:[[0,.1,0,.9],[0,-.2,0,0],[0,.75,0,0],[0,-.8,0,0]],meta:{role:'source'}},
  {meta:{role:'source'}},{}];
nodes['contribution-plot'].layout={shapes:[]};
nodes['contribution-plot'].on=(name,fn)=>{events[name]=fn;};
const document={getElementById:id=>nodes[id],querySelectorAll:()=>buttons,createElement:()=>element()};
const Plotly={relayout:()=>{}};
const fakeViewer={removeAllLabels(){},setStyle(selection,style){styles.push({selection,style});},
  addStyle(){},addLabel(){},addModel(){},zoomTo(){},render(){}};
const $3Dmol={createViewer:()=>fakeViewer};
"""
        result = evaluate_javascript(fixture + runner.LINKED_VIEWER_JS + r"""
events.plotly_click({points:[{curveNumber:0,x:2,y:1}]});
const first={source:selectedKey,count:nodes['receiver-rows'].children.length,
  receiver3:buttons[2].classList.values.has('selected-receiver'),
  receiver4:buttons[3].classList.values.has('selected-receiver'),
  status:nodes['selection-status'].textContent,styles:styles.slice()};
nodes.threshold.value='1';nodes.threshold.listeners.input();
const empty=nodes['receiver-rows'].children.length;
nodes.threshold.value='.5';buttons[3].listeners.click();
events.plotly_click({points:[{curveNumber:2,x:2}]});
const sequenceClick={source:selectedKey,count:nodes['receiver-rows'].children.length};
nodes['clear-selection'].listeners.click();
JSON.stringify({first,empty,sequenceClick,cleared:selectedKey===null&&buttons.every(b=>b.classList.values.size===0)});
""")
        self.assertEqual(result["first"]["source"], 2)
        self.assertEqual(result["first"]["count"], 2)
        self.assertTrue(result["first"]["receiver3"])
        self.assertTrue(result["first"]["receiver4"])
        self.assertIn("1/2 receivers have mapped coordinates", result["first"]["status"])
        selected_atoms = [s["selection"]["serial"] for s in result["first"]["styles"] if "serial" in s["selection"]]
        self.assertEqual(selected_atoms, [[104], [102]])
        self.assertEqual(result["empty"], 0)
        self.assertEqual(result["sequenceClick"], {"source": 4, "count": 1})
        self.assertTrue(result["cleared"])

    def test_full_sequence_report_keeps_positions_beyond_512(self):
        with tempfile.TemporaryDirectory() as tmp:
            n = 513
            matrix = np.zeros((n, n)); matrix[-1, -1] = .123
            runner.write_linked_report(Path(tmp), "A"*n, np.full((n, 2), .5), matrix,
                                       matrix.mean(axis=0), [], True)
            page = (Path(tmp) / "report.html").read_text()
            match = re.search(r'Plotly\.newPlot\(\s*"contribution-plot",\s*', page)
            traces, _ = json.JSONDecoder().raw_decode(page[match.end():])
            self.assertEqual(len(traces[0]["z"]), n)
            self.assertEqual(traces[0]["z"][-1][-1], .123)
            self.assertIn('data-pos="513"', page)
            self.assertNotIn('first 512', page)

    def test_linear_baseline_does_not_offer_contribution_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner.write_linked_report(Path(tmp), "ACD", np.full((3, 2), .5), None, None, [], False)
            page = (Path(tmp) / "report.html").read_text()
            self.assertIn('<section hidden>', page)
            self.assertIn('has no task-specific contribution matrix', page)

    def test_structure_mapping_preserves_missing_residues_and_insertion_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequence = "ACDEFGHIKLMNPQ"
            names = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE",
                     "LEU", "MET", "ASN", "PRO", "GLN"]
            lines = []
            for index, name in enumerate(names):
                insertion = "A" if index == 4 else " "
                lines.append(f"ATOM  {index+1:5d}  CA  {name} A{100+index:4d}{insertion}   "
                             f"{index*3.8:8.3f}{0:8.3f}{0:8.3f}{1:6.2f}{20:6.2f}           C  \n")
            pdb = root / "input.pdb"
            pdb.write_text("".join(lines) + "TER\nEND\n")
            (root / "run.json").write_text(json.dumps({"sequence": sequence, "bands": [
                {"sign": 1, "start": 4, "end": 5, "peak_position": 5},
                {"sign": -1, "start": 9, "end": 9, "peak_position": 9}]}))
            matrix = np.zeros((len(sequence), len(sequence)))
            np.savez(root / "contributions.npz", mean_C=matrix, mean_I=matrix.mean(axis=0),
                     mean_probabilities=np.full((len(sequence), 2), .5))
            # The display labels are not needed by the structure mapper, but are
            # used when rebuilding the integrated contribution report.
            summary = json.loads((root / "run.json").read_text())
            for band in summary["bands"]:
                band.update(peak_amino_acid=sequence[band["peak_position"]-1],
                            class_support="multi-PB" if band["sign"]>0 else "single-PB")
            (root / "run.json").write_text(json.dumps(summary))
            runner.map_structure(root, pdb, "A")
            result = json.loads((root / "structure_mapping.json").read_text())
            self.assertEqual(result["identity"], 1)
            self.assertFalse(result["mapping"][8]["resolved"])
            self.assertEqual(result["mapping"][4]["pdb_residue"], 104)
            self.assertEqual(result["mapping"][4]["insertion_code"], "A")
            self.assertIn("F104A", (root / "structure.html").read_text())
            page = (root / "report.html").read_text()
            payload = json.loads(re.search(r'<script id="viewer-data" type="application/json">(.*?)</script>', page).group(1))
            self.assertFalse(payload["structure"]["mapping"][8]["resolved"])
            self.assertNotIn("9", payload["structure"]["atom_serials"])
            self.assertIn("structure-viewer", page)


if __name__ == "__main__":
    unittest.main()
