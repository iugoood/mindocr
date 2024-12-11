from typing import Optional, Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, mint

from ..utils import GRUCell

__all__ = ["AttentionHead"]


class AttentionHead(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int = 256,
        batch_max_length: int = 25,
    ) -> None:
        """Attention head, based on
        `"Robust text recognizer with Automatic REctification"
        <https://arxiv.org/abs/1603.03915/>`_.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of the output channels.
            hidden_size: Hidden size in the attention cell. Default: 256.
            batch_max_length: The maximum length of the output. Default: 25.
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_classes = out_channels
        self.batch_max_length = batch_max_length

        self.attention_cell = AttentionCell(
            self.in_channels, self.hidden_size, self.num_classes
        )
        self.generator = mint.nn.Linear(hidden_size, self.num_classes)

        self.one = Tensor(1.0, ms.float32)
        self.zero = Tensor(0.0, ms.float32)

        self.argmax = ops.Argmax(axis=1)

    def _char_to_onehot(self, input_char: Tensor, onehot_dim: int) -> Tensor:
        input_one_hot = ops.one_hot(input_char, onehot_dim, self.one, self.zero)
        return input_one_hot

    def construct(self, inputs: Tensor, targets: Optional[Tuple[Tensor, ...]] = None) -> Tensor:
        # convert the inputs from [W, BS, C] to [BS, W, C]
        inputs = mint.permute(inputs, (1, 0, 2))
        N = inputs.shape[0]
        num_steps = self.batch_max_length + 1  # for <STOP> symbol

        hidden = mint.zeros((N, self.hidden_size), dtype=inputs.dtype)

        if targets is not None:
            # training branch
            targets = targets[0]
            output_hiddens = list()
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets[:, i], self.num_classes)
                hidden, _ = self.attention_cell(hidden, inputs, char_onehots)
                output_hiddens.append(mint.unsqueeze(hidden, dim=1))
            output = mint.concat(output_hiddens, dim=1)
            probs = self.generator(output)
        else:
            # inference branch
            # <GO> symbol
            targets = ops.zeros((N,), ms.int32)
            probs = list()
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, self.num_classes)
                hidden, _ = self.attention_cell(hidden, inputs, char_onehots)
                probs_step = self.generator(hidden)
                probs.append(probs_step)
                next_input = self.argmax(probs_step)
                targets = next_input
            probs = mint.stack(probs, dim=1)
            probs = mint.nn.Softmax(dim=2)(probs)
        return probs


class AttentionCell(nn.Cell):
    def __init__(self, input_size: int, hidden_size: int, num_embeddings: int) -> None:
        super().__init__()
        self.i2h = mint.nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = mint.nn.Linear(hidden_size, hidden_size)
        self.score = mint.nn.Linear(hidden_size, 1, bias=False)
        self.rnn = GRUCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

        self.bmm = mint.bmm

    def construct(
        self, prev_hidden: Tensor, batch_H: Tensor, char_onehots: Tensor
    ) -> Tuple[Tensor, Tensor]:
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden)
        prev_hidden_proj = mint.unsqueeze(prev_hidden_proj, 1)

        res = mint.add(batch_H_proj, prev_hidden_proj)
        res = ops.tanh(res)
        e = self.score(res)

        alpha = mint.nn.Softmax(dim=1)(e)
        alpha = mint.permute(alpha, (0, 2, 1))
        context = mint.squeeze(self.bmm(alpha, batch_H), dim=1)
        concat_context = mint.concat([context, char_onehots], dim=1)

        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha
