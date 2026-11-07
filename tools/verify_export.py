"""Check that a 4-bit ONNX export of Audio8 TTS holds a given checkpoint.

    python tools/verify_export.py MODEL_SAFETENSORS OFFICIAL_DIR EXPORT_DIR

Edge0 publishes an ONNX build of Preview only; the add-on's Realistic model
is an independent export (see tools/pin_model.py). This compares the export's
two transformers with the official Preview graphs and with the checkpoint:

- graph by graph: the same inputs and outputs, opsets, nodes and node
  attributes, in the same order, and the same constants;
- each 4-bit matrix (MatMulNBits) against the checkpoint's tensor quantized
  again here with ONNX Runtime's own quantizer: byte for byte, scales and
  zero points included. The output head is the semantic rows of the tied
  embedding table, then the end-of-speech row;
- each 4-bit embedding table (GatherBlockQuantized) dequantized, against the
  checkpoint's: the mean error must be within half a 4-bit step. A loose
  check: it tells a wrong table from the right one, not Realistic's from
  Preview's (Preview's text embeddings are only a little further off, and
  the codebook embeddings are the same in both).

Needs onnx, onnxruntime, numpy and safetensors (not the add-on's runtime
dependencies). Exits 1 on any mismatch. The codec and tokenizer files are not
checked here: tools/pin_model.py requires them to be the official files.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.capi._pybind_state import quantize_matmul_4bits
from safetensors.numpy import load_file

GRAPHS = ('slow_ar_int4.onnx', 'fast_ar_int4.onnx')


def attributes(node) -> dict:
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute
            if a.type != onnx.AttributeProto.TENSOR}


def structure(model) -> list:
    return [(n.op_type, n.domain, sorted(attributes(n).items())) for n in model.graph.node]


def constants(model) -> list[bytes]:
    return [numpy_helper.to_array(a.t).tobytes() for n in model.graph.node if n.op_type == 'Constant'
            for a in n.attribute if a.type == onnx.AttributeProto.TENSOR]


def signature(model) -> list:
    return [(v.name, str(v.type)) for v in list(model.graph.input) + list(model.graph.output)]


def quantize(weight: np.ndarray, like_b, like_s, like_z, block: int) -> tuple:
    """ONNX Runtime's 4-bit asymmetric quantization of a [K, N] matrix."""
    k, n = weight.shape
    b, s, z = np.zeros_like(like_b), np.zeros_like(like_s), np.zeros_like(like_z)
    quantize_matmul_4bits(b, np.ascontiguousarray(weight.astype(like_s.dtype)), s, z, block, n, k, False)
    return b, s, z


def dequantize_table(data, scales, zeros, block) -> np.ndarray:
    blocks = scales.shape[1]
    q = data.astype(np.float32)
    if q.shape[1] * 2 == blocks * block:   # two values a byte
        q = np.stack([data & 0x0F, data >> 4], -1).reshape(data.shape[0], -1).astype(np.float32)
    if zeros is None:
        z = np.full(scales.shape, 8, np.float32)
    else:
        z = zeros.astype(np.float32)
        if z.shape[1] != blocks:
            z = np.stack([zeros & 0x0F, zeros >> 4], -1).reshape(zeros.shape[0], -1)[:, :blocks].astype(np.float32)
    s = scales.astype(np.float32)
    return ((q.reshape(q.shape[0], blocks, block) - z[..., None]) * s[..., None]).reshape(q.shape[0], -1)


def main() -> int:
    checkpoint, official, export = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    weights = load_file(str(checkpoint))
    config_path = checkpoint.with_name('config.json')
    config = json.loads(config_path.read_text()) if config_path.is_file() else {
        'semantic_begin_id': 151678, 'semantic_end_id': 155773, 'eos_token_id': 151645}
    head_rows = np.r_[np.arange(config['semantic_begin_id'], config['semantic_end_id'] + 1),
                      config['eos_token_id']]
    by_shape: dict = {}
    for name, tensor in weights.items():
        if tensor.ndim == 2:
            by_shape.setdefault(tensor.T.shape, []).append(name)
    failures = 0
    for graph in GRAPHS:
        ref = onnx.load(str(official / graph))
        new = onnx.load(str(export / graph))
        same = (signature(ref) == signature(new) and structure(ref) == structure(new)
                and constants(ref) == constants(new)
                and [(o.domain, o.version) for o in ref.opset_import] == [(o.domain, o.version) for o in new.opset_import])
        print(f'{graph}: {len(new.graph.node)} nodes, same graph as the official one: {same}')
        failures += not same
        init = {i.name: numpy_helper.to_array(i) for i in new.graph.initializer}
        exact = total = 0
        for node in new.graph.node:
            if node.op_type == 'MatMulNBits':
                total += 1
                a = attributes(node)
                b, s, z = (init[name] for name in node.input[1:4])
                if node.name == '/MatMul_Q4':   # the tied output head
                    candidates = [weights['embeddings.weight'][head_rows]]
                else:
                    candidates = [weights[name] for name in by_shape.get((a['K'], a['N']), [])]
                if any(all(np.array_equal(x, y) for x, y in zip(quantize(c.astype(np.float32).T, b, s, z, a['block_size']), (b, s, z)))
                       for c in candidates):
                    exact += 1
                else:
                    print(f'  MISMATCH {node.name}')
            elif node.op_type == 'GatherBlockQuantized':
                a = attributes(node)
                data, scales = init[node.input[0]], init[node.input[2]]
                zeros = init[node.input[3]] if len(node.input) > 3 and node.input[3] else None
                table = dequantize_table(data, scales, zeros, a.get('block_size', 128))
                name = min((n for n, t in weights.items() if t.shape == table.shape),
                           key=lambda n: float(np.abs(table - weights[n].astype(np.float32)).mean()))
                error = float(np.abs(table - weights[name].astype(np.float32)).mean())
                step = float(np.abs(scales.astype(np.float32)).mean())
                ok = error < step / 2
                print(f'  {node.name}: {name}, mean error {error:.5f} (half a step: {step / 2:.5f})')
                failures += not ok
        print(f'  4-bit matrices byte-identical to the checkpoint requantized: {exact}/{total}')
        failures += exact != total
    print('OK' if not failures else f'FAILED ({failures})')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
