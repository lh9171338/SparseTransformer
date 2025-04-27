import unittest
import logging
import warnings
warnings.filterwarnings("ignore")
import torch
import sys 
sys.path.append("..") 
import sptr
from sptr.utils import scatter_softmax_csr


class Test(unittest.TestCase):
    """Test"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = torch.load("data.pth")

    @staticmethod
    def test_attention(
        query,
        key,
        value,
        index_0,
        index_0_offsets,
        index_1,
        index_1_offsets,
        n_max,
        dtype=torch.float32,
        **kwargs,
    ):
        query = query.to(dtype).detach().requires_grad_()
        key = key.to(dtype).detach().requires_grad_()
        value = value.to(dtype).detach().requires_grad_()

        attn = sptr.attention_step1(
            query,
            key,
            index_0,
            index_0_offsets,
            index_1,
            index_1_offsets,
            n_max,
        )
        attn = scatter_softmax_csr(src=attn, indptr=index_0_offsets.long(), dim=0)
        output = sptr.attention_step2(
            attn,
            value,
            index_0,
            index_0_offsets,
            index_1,
            index_1_offsets,
            n_max,
        )
        loss = output.mean()
        loss.backward()

        result = dict(
            output=output,
            query_grad=query.grad.clone(),
            key_grad=key.grad.clone(),
            value_grad=value.grad.clone(),
        )
        query.grad.zero_()
        key.grad.zero_()
        value.grad.zero_()

        print(f"max_memory_allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.0f}MB")
        print(f"max_memory_reserved: {torch.cuda.max_memory_reserved() / (1024 ** 2):.0f}MB")

        return result

    @staticmethod
    def test_attention_with_rel_pos(
        query,
        key,
        value,
        index_0,
        index_0_offsets,
        index_1,
        index_1_offsets,
        relative_pos_query_table,
        relative_pos_key_table,
        relative_pos_value_table,
        relative_position_index,
        n_max,
        dtype=torch.float32,
        **kwargs,
    ):
        query = query.to(dtype).detach().requires_grad_()
        key = key.to(dtype).detach().requires_grad_()
        value = value.to(dtype).detach().requires_grad_()
        relative_pos_query_table = relative_pos_query_table.to(dtype).detach().requires_grad_()
        relative_pos_key_table = relative_pos_key_table.to(dtype).detach().requires_grad_()
        relative_pos_value_table = relative_pos_value_table.to(dtype).detach().requires_grad_()

        attn = sptr.dot_prod_with_idx_all(
            query,
            index_0,
            index_0_offsets,
            key,
            index_1,
            index_1_offsets,
            relative_pos_query_table,
            relative_pos_key_table,
            relative_position_index,
            n_max,
        )
        attn = scatter_softmax_csr(src=attn, indptr=index_0_offsets.long(), dim=0)
        output = sptr.attention_step2_with_rel_pos_value(
            attn,
            value,
            index_0,
            index_0_offsets,
            n_max,
            index_1,
            index_1_offsets,
            relative_pos_value_table,
            relative_position_index,
        )
        loss = output.mean()
        loss.backward()

        result = dict(
            output=output,
            query_grad=query.grad.clone(),
            key_grad=key.grad.clone(),
            value_grad=value.grad.clone(),
            relative_pos_query_table_grad=relative_pos_query_table.grad.clone(),
            relative_pos_key_table_grad=relative_pos_key_table.grad.clone(),
            relative_pos_value_table_grad=relative_pos_value_table.grad.clone(),
        )
        query.grad.zero_()
        key.grad.zero_()
        value.grad.zero_()
        relative_pos_query_table.grad.zero_()
        relative_pos_key_table.grad.zero_()
        relative_pos_value_table.grad.zero_()

        print(f"max_memory_allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.0f}MB")
        print(f"max_memory_reserved: {torch.cuda.max_memory_reserved() / (1024 ** 2):.0f}MB")

        return result

    @staticmethod
    def compare(result1, result2):
        for name, value1 in result1.items():
            value2 = result2[name]
            diff = (value1 - value2).abs()
            print(f"{name}, max: {diff.max():.1e}, mean: {diff.mean():.1e}")

    def test_attention_fp32(self):
        self.test_attention(**self.data, dtype=torch.float32)

    def test_attention_fp16(self):
        self.test_attention(**self.data, dtype=torch.float16)

    def test_attention_bf16(self):
        try:
            self.test_attention(**self.data, dtype=torch.bfloat16)
        except RuntimeError as e:
            logging.warnings("BF16 is not supported")

    def test_attention_with_rel_pos_fp32(self):
        self.test_attention_with_rel_pos(**self.data, dtype=torch.float32)

    def test_attention_with_rel_pos_fp16(self):
        self.test_attention_with_rel_pos(**self.data, dtype=torch.float16)   

    def test_attention_with_rel_pos_bf16(self):
        try:
            self.test_attention_with_rel_pos(**self.data, dtype=torch.bfloat16)
        except RuntimeError as e:
            logging.warning("BF16 is not supported")

    def test_attention_precision(self):
        result_fp32 = self.test_attention(**self.data, dtype=torch.float32)
        result_fp16 = self.test_attention(**self.data, dtype=torch.float16)
        self.compare(result_fp32, result_fp16)
        try:
            result_bf16 = self.test_attention(**self.data, dtype=torch.bfloat16)
            self.compare(result_fp32, result_bf16)
        except RuntimeError as e:
            logging.warning("BF16 is not supported")

    def test_attention_with_rel_pos_precision(self):
        result_fp32 = self.test_attention_with_rel_pos(**self.data, dtype=torch.float32)
        result_fp16 = self.test_attention_with_rel_pos(**self.data, dtype=torch.float16)
        self.compare(result_fp32, result_fp16)
        try:
            result_bf16 = self.test_attention_with_rel_pos(**self.data, dtype=torch.bfloat16)
            self.compare(result_fp32, result_bf16)
        except RuntimeError as e:
            logging.warning("BF16 is not supported")


if __name__ == "__main__":
    # set base logging config
    fmt = "[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)

    unittest.main()
