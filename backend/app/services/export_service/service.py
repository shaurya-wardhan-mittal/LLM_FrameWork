"""Model export: LoRA, merged, GGUF, HuggingFace Hub."""

from pathlib import Path
from typing import Any


class ExportService:
    @staticmethod
    def save_lora(model, tokenizer, output_dir: Path) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        return str(output_dir)

    @staticmethod
    def save_merged(model, tokenizer, output_dir: Path) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_merged(str(output_dir), tokenizer, save_method="merged_16bit")
        return str(output_dir)

    @staticmethod
    def save_gguf(model, tokenizer, output_dir: Path, quantization: str = "q4_k_m") -> str:
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_gguf(str(output_dir), tokenizer, quantization_method=quantization)
        return str(output_dir)

    def export(
        self,
        *,
        model,
        tokenizer,
        export_format: str,
        output_dir: Path,
        quantization: str = "q4_k_m",
        push_to_hub: bool = False,
        hf_repo_id: str | None = None,
        hf_token: str | None = None,
    ) -> dict[str, Any]:
        if export_format == "lora":
            path = self.save_lora(model, tokenizer, output_dir / "lora")
        elif export_format == "merged":
            path = self.save_merged(model, tokenizer, output_dir / "merged")
        elif export_format == "gguf":
            path = self.save_gguf(model, tokenizer, output_dir / "gguf", quantization)
        elif export_format == "hf":
            path = self.save_merged(model, tokenizer, output_dir / "hf")
            if push_to_hub and hf_repo_id:
                model.push_to_hub(hf_repo_id, token=hf_token)
                tokenizer.push_to_hub(hf_repo_id, token=hf_token)
        else:
            raise ValueError(f"Unknown export format: {export_format}")

        return {
            "export_format": export_format,
            "storage_path": path,
            "hf_repo_id": hf_repo_id if push_to_hub else None,
        }
