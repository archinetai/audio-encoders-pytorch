import sys

import torch.nn as nn

REVISION = {"autoencoder1d-AT-v1": "57b6cde1969208d10fdd3e813708c1abe49f25c1"}


class AudioEncoders:
    @staticmethod
    def from_pretrained(name: str) -> nn.Module:
        assert_message = "Pretrained AudioEncoders requires `pip install transformers`"
        assert "transformers" in sys.modules, assert_message
        from transformers import AutoModel

        return AutoModel.from_pretrained(
            f"archinetai/{name}", trust_remote_code=True, revision=REVISION[name]
        )
