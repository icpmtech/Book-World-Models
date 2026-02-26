# Chapter 15

## Ontology-Driven World Models: From Palantir Foundry to LLM-Integrated Intelligence

The preceding chapters built a complete architecture for financial World Models — from latent-state inference through probabilistic price prediction. This chapter introduces a complementary dimension: **structured knowledge**. Rather than letting a model learn all structure from raw data, we can encode domain knowledge explicitly as an **ontology** and use it to ground both the World Model's state space and the reasoning of LLM agents that interact with it.

The inspiration comes from enterprise-scale data platforms — most notably **Palantir Foundry** — which proved that a rigorously maintained ontology of business objects and their relationships dramatically accelerates both human and automated decision-making. Applying the same philosophy to financial World Models yields a new class of system: the **Ontology-Driven World Model (ODWM)**.

---

## What Is an Ontology in This Context?

An **ontology** is a formal, machine-readable specification of:

- **Classes** (entity types) — `Asset`, `Company`, `MarketEvent`, `EconomicIndicator`, `Portfolio`, `RiskFactor`
- **Properties** — attributes of each class (`ticker`, `sector`, `country`, `currency`)
- **Relationships** — typed edges between classes (`Asset issuedBy Company`, `Company operatesIn Sector`, `MarketEvent impacts Asset`)
- **Constraints** — rules governing valid states (`Portfolio.weights sum to 1.0`)
- **Temporal semantics** — how entities and relationships change over time

Unlike a flat database schema, an ontology supports **inference**: given that a `Company` operates in the `Technology` sector and the `Technology` sector is correlated with the `NASDAQ` index, a reasoning system can infer that the company's price is likely correlated with the index — even without an explicit `correlatedWith` edge.

```
Ontology Schema (simplified OWL/RDF-like notation):

Class: Asset
  properties: ticker (string), assetClass (enum), currency (string)
  relations:  issuedBy → Company, denominatedIn → Currency

Class: Company
  properties: name (string), sector (Sector), country (Country)
  relations:  operatesIn → Sector, headquarteredIn → Country

Class: MarketEvent
  properties: eventType (enum), timestamp (datetime), severity (float)
  relations:  impacts → Asset (weight: float), causedBy → EconomicIndicator

Class: Portfolio
  properties: name (string), riskTarget (float), currency (string)
  relations:  holds → Asset (weight: float), managedBy → Agent
```

---

## The Palantir Foundry Paradigm

Palantir Foundry introduced a landmark concept in enterprise AI: **the Ontology as the operating system for data**. Key principles of the Foundry paradigm that apply to financial World Models are:

### 1. Objects, Not Rows

Instead of tables, Foundry organises data as **objects** with identity, properties, and relationships. An `Asset` object is not a row in a database — it is a first-class entity that can be queried, linked, analysed, and simulated.

### 2. Semantic Layer

All data access goes through the ontology. Analysts do not write SQL — they query objects and relationships in domain language. An LLM can do the same, using the ontology as a grounding interface.

### 3. Action Templates

The ontology defines not just knowledge but **actions**: operations that can be taken on objects and their expected effects. `Rebalance(portfolio, target_weights)` is an action template with pre-conditions and post-conditions defined in the ontology.

### 4. Live Objects

Foundry objects are not static snapshots. They update in real time as underlying data changes. A `Portfolio` object's `currentValue` property reflects live prices; its `riskMetrics` update whenever its holdings or market conditions change.

---

## Architecture: Ontology-Driven World Model

The Ontology-Driven World Model (ODWM) integrates four layers:

### Layer 1 — Ontology (Knowledge Graph)

The ontology encodes the financial domain's entity structure, relationships, and constraints. It serves as:

- A **grounding interface** for LLM agents (preventing hallucinated entities)
- A **structural prior** for the World Model's latent space
- An **action space definition** for the Controller component
- A **validation layer** for generated outputs (e.g., portfolio weights must sum to 1)

### Layer 2 — LLM Integration

LLMs interact with the ontology through two mechanisms:

**Ontology-Aware Query Engine**
```python
class OntologyQueryEngine:
    def __init__(self, ontology: KnowledgeGraph, llm: LargeLanguageModel):
        self.graph = ontology
        self.llm   = llm

    def query(self, natural_language_question: str) -> OntologyQueryResult:
        intent   = self.llm.extract_intent(
            question=natural_language_question,
            entity_types=self.graph.class_names(),
        )
        grounded = self.graph.ground_entities(intent.entities)
        result   = self.graph.traverse(
            start_nodes=grounded,
            relation_path=intent.relation_path,
            filters=intent.filters,
        )
        return self.llm.synthesise(result, question=natural_language_question)
```

**Semantic Embedding Alignment**
```python
class OntologyEmbedder:
    def embed_entity(self, entity: OntologyObject) -> Tensor:
        description = self._build_description(entity)
        return self.llm_encoder(description)

    def _build_description(self, entity: OntologyObject) -> str:
        props = ", ".join(f"{k}={v}" for k, v in entity.properties.items())
        rels  = ", ".join(
            f"{r.type} {r.target.name}" for r in entity.outgoing_relations
        )
        return f"{entity.class_name} '{entity.name}': {props}. Relations: {rels}."
```

### Layer 3 — LLM Agent Orchestration

LLM agents use the ontology and the World Model together to reason, plan, and act.

### Layer 4 — World Model Core

The standard Vision–Memory–Controller architecture, augmented with ontology-aware encodings and semantically grounded action spaces.

---

## LLM Agents in Ontology-Driven World Models

LLM agents in an ODWM operate in a **five-stage loop**: Perceive → Reason → Simulate → Act → Observe.

### Stage 1 — Perceive

```python
class PerceptionModule:
    def perceive(self, raw_observation, ontology):
        z_market = self.world_model.encoder(raw_observation)
        active_entities = ontology.get_entities_at(raw_observation.timestamp)
        ontology_context = self.ontology_embedder.embed_batch(active_entities)
        z_fused = self.fusion_layer(
            torch.cat([z_market, ontology_context.mean(dim=0)])
        )
        return AgentPerception(latent_state=z_fused, entities=active_entities)
```

### Stage 2 — Reason

The LLM agent queries the ontology to ground and validate hypotheses before generating candidate actions consistent with ontology constraints.

### Stage 3 — Simulate

```python
class SimulationModule:
    def simulate(self, plan, world_model, horizon_days=21, n_paths=2000):
        outcomes = {}
        for action in plan.candidate_actions:
            a_encoded = self.action_encoder(action)
            latent_paths = world_model.dynamics.rollout(
                z_0=plan.latent_state,
                horizon=horizon_days,
                n_samples=n_paths,
                action=a_encoded,
            )
            price_dist = world_model.decoder(latent_paths)
            outcomes[action.id] = self.evaluate_portfolio(price_dist, action)
        return SimulationResult(action_outcomes=outcomes)
```

### Stage 4 — Act

The agent selects the best risk-adjusted action, validated against ontology constraints before execution.

### Stage 5 — Observe

```python
class ObservationModule:
    def observe(self, executed, actual_outcome, ontology, world_model):
        ontology.record_event(
            event_type='ActionOutcome',
            action=executed,
            outcome=actual_outcome,
            timestamp=actual_outcome.timestamp,
        )
        world_model.dynamics.update(
            z_prev=executed.latent_state_before,
            action=executed.encoded_action,
            z_next=world_model.encoder(actual_outcome.observation),
        )
```

---

## Multi-Agent Ontology-Driven Systems

Complex financial operations benefit from multiple specialised agents coordinated through a shared ontology:

| Agent Type | Ontology Role | World Model Role |
|---|---|---|
| **Analyst Agent** | Queries entity properties and relations | Interprets latent state as market regime |
| **Portfolio Agent** | Reads portfolio objects and weight constraints | Simulates portfolio outcomes across scenarios |
| **Risk Agent** | Monitors constraint violations and exposure limits | Evaluates tail risk of proposed actions |
| **Execution Agent** | Maps actions to order objects | Models market impact in the latent space |
| **Monitoring Agent** | Updates ontology with observed events | Detects latent state drift and regime changes |

---

## Ontology-Enhanced Latent State Representation

A key technical contribution of the ODWM is using the ontology to **structure the latent space** of the World Model.

### Typed Latent Dimensions

Instead of an unstructured latent vector `z_t ∈ ℝ^d`, the ODWM partitions the latent space according to ontology classes:

```
z_t = [z_assets | z_macro | z_events | z_relations | z_regime]
```

### Ontology-Constrained Dynamics

```python
class OntologyConstrainedDynamics(nn.Module):
    def step(self, z_t, a_t=None):
        h_next = self.gru(z_t.to_tensor(), self.h_prev)
        z_next_raw = self.projection(h_next)
        z_next = StructuredLatentState.from_tensor(z_next_raw, self.ontology_schema)
        z_next = self.ontology_schema.project_to_feasible(z_next)
        if a_t is not None:
            affected = self.ontology_schema.get_affected_partitions(a_t)
            for p in affected:
                z_next[p] = z_next[p] + self.action_heads[p](a_t)
        return z_next
```

---

## Palantir Foundry Concepts Mapped to Financial World Models

| Foundry Concept | ODWM Equivalent | Financial Example |
|---|---|---|
| **Object Type** | Ontology Class | `Asset`, `Portfolio`, `Company` |
| **Object Property** | Typed attribute | `Asset.volatility_30d`, `Portfolio.nav` |
| **Link Type** | Ontology Relation | `Asset issuedBy Company` |
| **Action Type** | World Model action | `Rebalance(portfolio, weights)` |
| **Pipeline** | World Model rollout | Latent trajectory z_t → z_{t+H} |
| **Contour (view)** | Regime-conditioned projection | Bull-market view of portfolio risk |
| **Workshop (app)** | Agent-facing interface | Portfolio management dashboard |
| **AIP (AI Platform)** | LLM + Ontology integration | Analyst agent querying market context |
| **Branch** | Counterfactual scenario | "What if rates rise 100bps?" simulation |

---

## Chapter Summary

- **Ontologies** provide a formal, machine-readable representation of domain knowledge — entities, relationships, constraints, and action templates — that goes far beyond what a neural network can learn implicitly from data alone
- The **Palantir Foundry paradigm** — objects, semantic layers, action templates, and live updating — translates directly into a powerful architecture for financial World Models
- **LLM-integrated ontologies** enable natural language access to structured knowledge, preventing hallucination by grounding LLM reasoning in verified entities and relationships
- **LLM agents** operating within an Ontology-Driven World Model follow a five-stage loop — Perceive, Reason, Simulate, Act, Observe — each stage enriched by both ontology context and World Model dynamics
- **Multi-agent coordination** through a shared ontology enables specialised agents (Analyst, Portfolio, Risk, Execution, Monitoring) to operate with complementary capabilities while maintaining a consistent knowledge state
- **Ontology-constrained latent spaces** make World Model representations interpretable and ensure that simulated trajectories respect domain constraints
- The combined ODWM architecture represents the state of the art for **explainable, grounded, and agentic financial AI**

---

## Looking Ahead

The Ontology-Driven World Model closes the loop on the architecture presented throughout this book. Raw market observations flow through an encoder into a structured latent space, dynamics models simulate forward trajectories, and now LLM agents — grounded in a rich knowledge graph — reason about those trajectories and act upon them with semantic awareness.

> *"The ontology is not a constraint on intelligence. It is the scaffolding on which intelligence builds."*
>
> A World Model without structured knowledge can simulate; one with an ontology can **understand**.
