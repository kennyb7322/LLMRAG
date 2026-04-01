"""
CLOUD PROVIDERS MODULE
=======================
Unified abstraction for Azure, AWS, GCP, and OCI cloud AI services.
Covers compute, storage, AI/ML platforms, and managed LLM endpoints.

Supported Platforms:
  - Azure: OpenAI Service, Copilot Studio, AI Foundry, Cognitive Services
  - AWS: Bedrock, SageMaker, Titan, Comprehend
  - GCP: Vertex AI, Gemini API, Cloud Natural Language, BigQuery ML
  - OCI: Generative AI Service, Data Science, Language Service
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from src.utils.logger import log


# ── Base Cloud Provider ─────────────────────────────────────────────────────

@dataclass
class CloudConfig:
    """Configuration for a cloud provider."""
    provider: str = ""
    region: str = ""
    credentials: Dict[str, str] = field(default_factory=dict)
    services: Dict[str, Any] = field(default_factory=dict)


class CloudProvider(ABC):
    """Abstract base class for cloud AI service integration."""

    def __init__(self, config: dict):
        self.config = config
        self.region = config.get("region", "")
        self._initialized = False

    @abstractmethod
    def initialize(self): ...

    @abstractmethod
    def call_llm(self, prompt: str, model: str = "", **kwargs) -> str: ...

    @abstractmethod
    def embed_text(self, texts: List[str], model: str = "") -> List[List[float]]: ...

    @abstractmethod
    def health_check(self) -> Dict[str, Any]: ...


# ── AZURE ───────────────────────────────────────────────────────────────────

class AzureProvider(CloudProvider):
    """
    Microsoft Azure AI Integration
    ================================
    Services:
      - Azure OpenAI Service (GPT-4o, GPT-4-Turbo, Embeddings)
      - Microsoft Copilot Studio (agent orchestration)
      - Azure AI Foundry / Frontier Models (Phi-4, MAI-1)
      - Azure Cognitive Services (Language, Vision, Speech)
      - Azure AI Search (vector + semantic hybrid search)
      - Azure Machine Learning (custom model hosting)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.endpoint = config.get("endpoint", "") or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        self.api_key = config.get("api_key", "") or os.environ.get("AZURE_OPENAI_API_KEY", "")
        self.api_version = config.get("api_version", "2024-12-01-preview")
        self.deployment = config.get("deployment", "gpt-4o")

        # Copilot Studio config
        self.copilot = config.get("copilot_studio", {})
        # AI Foundry / Frontier
        self.frontier = config.get("frontier", {})
        # AI Search
        self.ai_search = config.get("ai_search", {})

    def initialize(self):
        if not self.endpoint:
            log.warning("Azure: No endpoint configured")
            return
        try:
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
            )
            self._initialized = True
            log.info(f"Azure OpenAI initialized: {self.endpoint}")
        except ImportError:
            log.error("openai package not installed")

    def call_llm(self, prompt: str, model: str = "", **kwargs) -> str:
        if not self._initialized:
            self.initialize()
        try:
            response = self._client.chat.completions.create(
                model=model or self.deployment,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 4096),
            )
            return response.choices[0].message.content
        except Exception as e:
            log.error(f"Azure LLM error: {e}")
            return f"Error: {e}"

    def embed_text(self, texts: List[str], model: str = "") -> List[List[float]]:
        if not self._initialized:
            self.initialize()
        try:
            response = self._client.embeddings.create(
                model=model or "text-embedding-3-small",
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            log.error(f"Azure embedding error: {e}")
            return [[0.0] * 1536] * len(texts)

    def call_copilot_studio(self, message: str, agent_id: str = "") -> Dict[str, Any]:
        """Invoke a Microsoft Copilot Studio agent."""
        bot_endpoint = self.copilot.get("endpoint", "")
        bot_secret = self.copilot.get("secret", "") or os.environ.get("COPILOT_STUDIO_SECRET", "")
        if not bot_endpoint:
            log.warning("Copilot Studio: No endpoint configured")
            return {"error": "No endpoint"}
        try:
            import requests
            resp = requests.post(
                f"{bot_endpoint}/api/messages",
                headers={"Authorization": f"Bearer {bot_secret}", "Content-Type": "application/json"},
                json={"type": "message", "text": message},
                timeout=30,
            )
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def call_frontier_model(self, prompt: str, model: str = "Phi-4") -> str:
        """Call Azure AI Foundry Frontier models (Phi-4, MAI-1, etc.)."""
        endpoint = self.frontier.get("endpoint", "")
        key = self.frontier.get("api_key", "") or os.environ.get("AZURE_FRONTIER_KEY", "")
        if not endpoint:
            log.warning("Frontier: No endpoint configured")
            return ""
        try:
            import requests
            resp = requests.post(
                f"{endpoint}/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}]},
                timeout=60,
            )
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            return f"Error: {e}"

    def search_ai_search(self, query: str, index: str = "", top_k: int = 10) -> List[Dict]:
        """Query Azure AI Search (vector + semantic hybrid)."""
        search_endpoint = self.ai_search.get("endpoint", "")
        search_key = self.ai_search.get("api_key", "") or os.environ.get("AZURE_SEARCH_KEY", "")
        index_name = index or self.ai_search.get("index", "llmrag-index")
        if not search_endpoint:
            return []
        try:
            import requests
            resp = requests.post(
                f"{search_endpoint}/indexes/{index_name}/docs/search?api-version=2024-07-01",
                headers={"api-key": search_key, "Content-Type": "application/json"},
                json={"search": query, "top": top_k, "queryType": "semantic", "semanticConfiguration": "default"},
                timeout=15,
            )
            return resp.json().get("value", [])
        except Exception as e:
            log.error(f"Azure AI Search error: {e}")
            return []

    def health_check(self) -> Dict[str, Any]:
        return {
            "provider": "azure",
            "endpoint": self.endpoint[:40] + "..." if self.endpoint else "not set",
            "deployment": self.deployment,
            "copilot_studio": bool(self.copilot.get("endpoint")),
            "frontier_models": bool(self.frontier.get("endpoint")),
            "ai_search": bool(self.ai_search.get("endpoint")),
            "initialized": self._initialized,
        }


# ── AWS ─────────────────────────────────────────────────────────────────────

class AWSProvider(CloudProvider):
    """
    Amazon Web Services AI Integration
    =====================================
    Services:
      - Amazon Bedrock (Claude, Llama, Titan, Mistral, Cohere)
      - Amazon SageMaker (custom model hosting)
      - Amazon Titan (embeddings, text generation)
      - Amazon Comprehend (NLP entity/sentiment)
      - Amazon Kendra (enterprise search)
      - Amazon Q (enterprise AI assistant)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.region = config.get("region", "us-east-1")
        self.bedrock_model = config.get("bedrock_model", "anthropic.claude-3-5-sonnet-20241022-v2:0")
        self.embed_model = config.get("embed_model", "amazon.titan-embed-text-v2:0")
        self.sagemaker = config.get("sagemaker", {})
        self.kendra = config.get("kendra", {})

    def initialize(self):
        try:
            import boto3
            self._bedrock = boto3.client("bedrock-runtime", region_name=self.region)
            self._initialized = True
            log.info(f"AWS Bedrock initialized: {self.region}")
        except ImportError:
            log.error("boto3 not installed — pip install boto3")

    def call_llm(self, prompt: str, model: str = "", **kwargs) -> str:
        if not self._initialized:
            self.initialize()
        model_id = model or self.bedrock_model
        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 4096),
                "temperature": kwargs.get("temperature", 0.1),
            })
            response = self._bedrock.invoke_model(modelId=model_id, body=body)
            result = json.loads(response["body"].read())
            return result.get("content", [{}])[0].get("text", "")
        except Exception as e:
            log.error(f"Bedrock error: {e}")
            return f"Error: {e}"

    def embed_text(self, texts: List[str], model: str = "") -> List[List[float]]:
        if not self._initialized:
            self.initialize()
        model_id = model or self.embed_model
        embeddings = []
        for text in texts:
            try:
                body = json.dumps({"inputText": text})
                response = self._bedrock.invoke_model(modelId=model_id, body=body)
                result = json.loads(response["body"].read())
                embeddings.append(result.get("embedding", [0.0] * 1536))
            except Exception as e:
                log.error(f"Titan embed error: {e}")
                embeddings.append([0.0] * 1536)
        return embeddings

    def call_sagemaker_endpoint(self, payload: dict, endpoint_name: str = "") -> dict:
        """Call a SageMaker hosted model endpoint."""
        name = endpoint_name or self.sagemaker.get("endpoint_name", "")
        if not name:
            return {"error": "No SageMaker endpoint configured"}
        try:
            import boto3
            sm = boto3.client("sagemaker-runtime", region_name=self.region)
            resp = sm.invoke_endpoint(
                EndpointName=name,
                ContentType="application/json",
                Body=json.dumps(payload),
            )
            return json.loads(resp["Body"].read())
        except Exception as e:
            return {"error": str(e)}

    def search_kendra(self, query: str, index_id: str = "") -> List[Dict]:
        """Query Amazon Kendra enterprise search."""
        idx = index_id or self.kendra.get("index_id", "")
        if not idx:
            return []
        try:
            import boto3
            kendra = boto3.client("kendra", region_name=self.region)
            resp = kendra.query(IndexId=idx, QueryText=query)
            return resp.get("ResultItems", [])
        except Exception as e:
            log.error(f"Kendra error: {e}")
            return []

    def health_check(self) -> Dict[str, Any]:
        return {
            "provider": "aws",
            "region": self.region,
            "bedrock_model": self.bedrock_model,
            "sagemaker": bool(self.sagemaker.get("endpoint_name")),
            "kendra": bool(self.kendra.get("index_id")),
            "initialized": self._initialized,
        }


# ── GCP ─────────────────────────────────────────────────────────────────────

class GCPProvider(CloudProvider):
    """
    Google Cloud Platform AI Integration
    =======================================
    Services:
      - Vertex AI (Gemini 2.5 Pro, Gemini Flash, PaLM, custom models)
      - Vertex AI Search (enterprise RAG search)
      - Cloud Natural Language API (entity/sentiment)
      - BigQuery ML (in-database ML)
      - Vertex AI Agents (agentic orchestration)
      - Model Garden (open model catalog)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.project_id = config.get("project_id", "") or os.environ.get("GCP_PROJECT_ID", "")
        self.location = config.get("location", "us-central1")
        self.model = config.get("model", "gemini-2.5-pro")
        self.vertex_search = config.get("vertex_search", {})

    def initialize(self):
        try:
            import google.generativeai as genai
            api_key = self.config.get("api_key", "") or os.environ.get("GOOGLE_API_KEY", "")
            if api_key:
                genai.configure(api_key=api_key)
                self._client = genai
                self._initialized = True
                log.info(f"GCP Gemini initialized: {self.model}")
            else:
                log.warning("GCP: No API key set")
        except ImportError:
            log.error("google-generativeai not installed — pip install google-generativeai")

    def call_llm(self, prompt: str, model: str = "", **kwargs) -> str:
        if not self._initialized:
            self.initialize()
        try:
            m = self._client.GenerativeModel(model or self.model)
            response = m.generate_content(prompt)
            return response.text
        except Exception as e:
            log.error(f"Gemini error: {e}")
            return f"Error: {e}"

    def embed_text(self, texts: List[str], model: str = "") -> List[List[float]]:
        if not self._initialized:
            self.initialize()
        try:
            embed_model = model or "models/text-embedding-004"
            result = self._client.embed_content(
                model=embed_model,
                content=texts,
                task_type="retrieval_document",
            )
            return result["embedding"] if isinstance(result["embedding"][0], list) else [result["embedding"]]
        except Exception as e:
            log.error(f"GCP embed error: {e}")
            return [[0.0] * 768] * len(texts)

    def search_vertex_ai_search(self, query: str) -> List[Dict]:
        """Query Vertex AI Search (formerly Enterprise Search)."""
        datastore_id = self.vertex_search.get("datastore_id", "")
        if not datastore_id:
            return []
        try:
            from google.cloud import discoveryengine_v1 as discoveryengine
            client = discoveryengine.SearchServiceClient()
            serving_config = (
                f"projects/{self.project_id}/locations/{self.location}"
                f"/dataStores/{datastore_id}/servingConfigs/default_config"
            )
            request = discoveryengine.SearchRequest(
                serving_config=serving_config, query=query, page_size=10
            )
            response = client.search(request)
            return [{"id": r.id, "document": str(r.document)} for r in response.results]
        except Exception as e:
            log.error(f"Vertex Search error: {e}")
            return []

    def health_check(self) -> Dict[str, Any]:
        return {
            "provider": "gcp",
            "project": self.project_id,
            "model": self.model,
            "vertex_search": bool(self.vertex_search.get("datastore_id")),
            "initialized": self._initialized,
        }


# ── OCI ─────────────────────────────────────────────────────────────────────

class OCIProvider(CloudProvider):
    """
    Oracle Cloud Infrastructure AI Integration
    =============================================
    Services:
      - OCI Generative AI Service (Cohere Command R+, Meta Llama)
      - OCI Data Science (custom model training/hosting)
      - OCI Language Service (NLP, entity extraction)
      - OCI Search with OpenSearch (vector + keyword)
      - OCI AI Agents
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.compartment_id = config.get("compartment_id", "") or os.environ.get("OCI_COMPARTMENT_ID", "")
        self.endpoint = config.get("endpoint", "")
        self.model = config.get("model", "cohere.command-r-plus")

    def initialize(self):
        try:
            import oci
            oci_config = oci.config.from_file() if not self.config.get("api_key") else {
                "tenancy": self.config.get("tenancy", ""),
                "user": self.config.get("user", ""),
                "key_file": self.config.get("key_file", ""),
                "fingerprint": self.config.get("fingerprint", ""),
                "region": self.region or "us-ashburn-1",
            }
            self._client = oci.generative_ai_inference.GenerativeAiInferenceClient(oci_config)
            self._initialized = True
            log.info("OCI Generative AI initialized")
        except ImportError:
            log.error("oci not installed — pip install oci")

    def call_llm(self, prompt: str, model: str = "", **kwargs) -> str:
        if not self._initialized:
            self.initialize()
        try:
            import oci
            details = oci.generative_ai_inference.models.ChatDetails(
                compartment_id=self.compartment_id,
                serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
                    model_id=model or self.model
                ),
                chat_request=oci.generative_ai_inference.models.CohereChatRequest(
                    message=prompt, max_tokens=kwargs.get("max_tokens", 4096)
                ),
            )
            response = self._client.chat(details)
            return response.data.chat_response.text
        except Exception as e:
            log.error(f"OCI LLM error: {e}")
            return f"Error: {e}"

    def embed_text(self, texts: List[str], model: str = "") -> List[List[float]]:
        log.warning("OCI embed: use Cohere embed via Bedrock or direct API")
        return [[0.0] * 1024] * len(texts)

    def health_check(self) -> Dict[str, Any]:
        return {
            "provider": "oci",
            "compartment": self.compartment_id[:20] + "..." if self.compartment_id else "not set",
            "model": self.model,
            "initialized": self._initialized,
        }


# ── Cloud Provider Factory ─────────────────────────────────────────────────

class CloudProviderFactory:
    """Factory to create and manage cloud provider instances."""

    PROVIDERS = {
        "azure": AzureProvider,
        "aws": AWSProvider,
        "gcp": GCPProvider,
        "oci": OCIProvider,
    }

    def __init__(self, config: dict):
        self.config = config
        self.providers: Dict[str, CloudProvider] = {}

    def initialize_all(self):
        """Initialize all configured cloud providers."""
        for name, provider_config in self.config.items():
            if name in self.PROVIDERS and provider_config.get("enabled", False):
                provider = self.PROVIDERS[name](provider_config)
                provider.initialize()
                self.providers[name] = provider
                log.info(f"Cloud provider '{name}' initialized")

    def get(self, name: str) -> Optional[CloudProvider]:
        return self.providers.get(name)

    def list_providers(self) -> List[str]:
        return list(self.providers.keys())

    def health_check_all(self) -> Dict[str, Any]:
        return {name: p.health_check() for name, p in self.providers.items()}
