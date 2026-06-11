from typing import Any

from app.config import get_settings
from app.services.model_service.catalog import CURATED_MODELS, estimate_params_from_id

settings = get_settings()


class ModelService:
    def list_catalog(self) -> list[dict[str, Any]]:
        return CURATED_MODELS

    def get_model_info(self, model_id: str) -> dict[str, Any]:
        for m in CURATED_MODELS:
            if m["id"] == model_id:
                return {**m, "estimated_vram_qlora_gb": m["params_b"] * 0.8}
        params = estimate_params_from_id(model_id)
        return {
            "id": model_id,
            "name": model_id.split("/")[-1],
            "params_b": params,
            "context_length": 8192,
            "family": "unknown",
            "estimated_vram_qlora_gb": params * 0.8,
        }

    def resolve(self, model_id: str) -> dict[str, Any]:
        info = self.get_model_info(model_id)
        info["unsloth_supported"] = True  # extend with HF API check in production
        info["hf_token_configured"] = bool(settings.hf_token)
        return info
