---
id: chapter-15
title: Ontology-Driven World Models — From Palantir Foundry to LLM-Integrated Intelligence
sidebar_label: "Chapter 15 — Ontology-Driven World Models"
sidebar_position: 16
---

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

Translating these four principles to financial World Models yields the architecture shown below.

---

## Architecture: Ontology-Driven World Model

![Ontology-Driven World Model Architecture](/img/ontology-world-model-architecture.svg)

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
    """
    Translates natural language questions into structured ontology queries.
    Grounds LLM reasoning in verified entities and relationships.
    """
    def __init__(self, ontology: KnowledgeGraph, llm: LargeLanguageModel):
        self.graph = ontology
        self.llm   = llm

    def query(self, natural_language_question: str) -> OntologyQueryResult:
        # 1. Parse intent and identify entity types mentioned
        intent = self.llm.extract_intent(
            question=natural_language_question,
            entity_types=self.graph.class_names(),
        )
        # 2. Ground entities: map names to verified ontology objects
        grounded = self.graph.ground_entities(intent.entities)
        # 3. Execute graph traversal
        result = self.graph.traverse(
            start_nodes=grounded,
            relation_path=intent.relation_path,
            filters=intent.filters,
        )
        # 4. LLM synthesises a natural language answer from graph result
        return self.llm.synthesise(result, question=natural_language_question)
```

**Semantic Embedding Alignment**
```python
class OntologyEmbedder:
    """
    Aligns ontology entity embeddings with LLM token embeddings.
    Enables the LLM to 'know' about entities it was not trained on.
    """
    def embed_entity(self, entity: OntologyObject) -> Tensor:
        # Structured description from ontology properties and relations
        description = self._build_description(entity)
        # Embed using the LLM's own encoder
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

The standard Vision–Memory–Controller architecture, now augmented with ontology-aware encodings and semantically grounded action spaces.

---

## LLM Agents in Ontology-Driven World Models

![LLM Agent — Ontology — World Model Loop](/img/llm-agent-ontology-loop.svg)

LLM agents in an ODWM operate in a **five-stage loop**:

### Stage 1 — Perceive

The agent receives observations from both the environment (raw market data) and the ontology (structured context about entities involved):

```python
class PerceptionModule:
    def perceive(
        self,
        raw_observation: MarketObservation,
        ontology: KnowledgeGraph,
    ) -> AgentPerception:
        # Encode raw market state
        z_market = self.world_model.encoder(raw_observation)

        # Retrieve relevant ontology context
        active_entities = ontology.get_entities_at(raw_observation.timestamp)
        ontology_context = self.ontology_embedder.embed_batch(active_entities)

        # Fuse: market state + ontology context
        z_fused = self.fusion_layer(
            torch.cat([z_market, ontology_context.mean(dim=0)])
        )
        return AgentPerception(latent_state=z_fused, entities=active_entities)
```

### Stage 2 — Reason

The LLM agent reasons about the current state, querying the ontology to resolve ambiguity:

```python
class ReasoningModule:
    def reason(
        self,
        perception: AgentPerception,
        query_engine: OntologyQueryEngine,
    ) -> AgentPlan:
        # LLM generates hypotheses about the current market state
        hypothesis = self.llm.generate_hypothesis(
            latent_context=perception.latent_state,
            entity_context=perception.entities,
        )

        # Ground and validate hypothesis against ontology
        validated = query_engine.validate_hypothesis(hypothesis)

        # Generate candidate actions consistent with ontology constraints
        candidate_actions = self.action_generator(
            validated_hypothesis=validated,
            allowed_actions=self.ontology.action_templates(),
        )
        return AgentPlan(hypothesis=validated, candidate_actions=candidate_actions)
```

### Stage 3 — Simulate

Before committing to an action, the agent runs the World Model forward to evaluate consequences:

```python
class SimulationModule:
    def simulate(
        self,
        plan: AgentPlan,
        world_model: WorldModel,
        horizon_days: int = 21,
        n_paths: int = 2000,
    ) -> SimulationResult:
        outcomes = {}
        for action in plan.candidate_actions:
            # Encode action in the World Model's action space
            a_encoded = self.action_encoder(action)

            # Roll out with the action applied
            latent_paths = world_model.dynamics.rollout(
                z_0=plan.latent_state,
                horizon=horizon_days,
                n_samples=n_paths,
                action=a_encoded,
            )
            price_dist = world_model.decoder(latent_paths)
            portfolio_outcomes = self.evaluate_portfolio(price_dist, action)
            outcomes[action.id] = portfolio_outcomes

        return SimulationResult(action_outcomes=outcomes)
```

### Stage 4 — Act

The agent selects the action with the best risk-adjusted expected outcome, validated against ontology constraints:

```python
class ActionModule:
    def act(
        self,
        simulation: SimulationResult,
        ontology: KnowledgeGraph,
    ) -> ExecutedAction:
        # Rank actions by Sharpe ratio of simulated outcomes
        ranked = sorted(
            simulation.action_outcomes.items(),
            key=lambda x: x[1].simulated_sharpe,
            reverse=True,
        )
        best_action_id, _ = ranked[0]
        best_action = simulation.action_outcomes[best_action_id].action

        # Validate against ontology constraints before execution
        violations = ontology.validate_action(best_action)
        if violations:
            raise OntologyConstraintViolation(violations)

        return self.executor.execute(best_action)
```

### Stage 5 — Observe

The agent observes the outcome, updates the ontology with new facts, and feeds the result back into the World Model's training loop:

```python
class ObservationModule:
    def observe(
        self,
        executed: ExecutedAction,
        actual_outcome: MarketOutcome,
        ontology: KnowledgeGraph,
        world_model: WorldModel,
    ) -> None:
        # Update ontology with observed outcome
        ontology.record_event(
            event_type='ActionOutcome',
            action=executed,
            outcome=actual_outcome,
            timestamp=actual_outcome.timestamp,
        )

        # Online update of World Model dynamics
        world_model.dynamics.update(
            z_prev=executed.latent_state_before,
            action=executed.encoded_action,
            z_next=world_model.encoder(actual_outcome.observation),
        )
```

---

## Multi-Agent Ontology-Driven Systems

Complex financial operations benefit from **multiple specialised agents** coordinated through a shared ontology:

### Agent Taxonomy

| Agent Type | Ontology Role | World Model Role |
|---|---|---|
| **Analyst Agent** | Queries entity properties and relations | Interprets latent state as market regime |
| **Portfolio Agent** | Reads portfolio objects and weight constraints | Simulates portfolio outcomes across scenarios |
| **Risk Agent** | Monitors constraint violations and exposure limits | Evaluates tail risk of proposed actions |
| **Execution Agent** | Maps actions to order objects | Models market impact in the latent space |
| **Monitoring Agent** | Updates ontology with observed events | Detects latent state drift and regime changes |

### Agent Coordination Protocol

```python
class OntologyAgentOrchestrator:
    """
    Coordinates multiple LLM agents sharing a common ontology and World Model.
    Implements a publish-subscribe pattern for inter-agent communication.
    """
    def __init__(self, ontology: KnowledgeGraph, world_model: WorldModel):
        self.ontology    = ontology
        self.world_model = world_model
        self.agents: dict[str, BaseAgent] = {}
        self.event_bus   = EventBus()

    def register_agent(self, name: str, agent: BaseAgent) -> None:
        self.agents[name] = agent
        agent.subscribe(self.event_bus)

    def run_cycle(self, observation: MarketObservation) -> CycleResult:
        # Parallel perception: all agents perceive the same observation
        perceptions = {
            name: agent.perceive(observation, self.ontology)
            for name, agent in self.agents.items()
        }

        # Sequential reasoning with shared ontology context
        analyst_plan   = self.agents['analyst'].reason(perceptions['analyst'])
        risk_assessment = self.agents['risk'].assess(
            perceptions['risk'],
            proposed_plan=analyst_plan,
        )

        # Simulation conditioned on risk constraints
        if risk_assessment.approved:
            simulation = self.agents['portfolio'].simulate(
                analyst_plan, self.world_model
            )
            action = self.agents['portfolio'].act(simulation, self.ontology)
            self.agents['execution'].execute(action)
        else:
            self.event_bus.publish('RiskVeto', risk_assessment)

        return CycleResult(analyst_plan, risk_assessment, simulation if risk_assessment.approved else None)
```

---

## Ontology-Enhanced Latent State Representation

A key technical contribution of the ODWM is using the ontology to **structure the latent space** of the World Model.

### Typed Latent Dimensions

Instead of an unstructured latent vector `z_t ∈ ℝ^d`, the ODWM partitions the latent space according to ontology classes:

```
z_t = [z_assets | z_macro | z_events | z_relations | z_regime]
       (n_assets × d_a) | (d_m) | (n_events × d_e) | (d_r) | (d_regime)
```

Each partition encodes a semantically coherent aspect of the market state, making the latent representation **interpretable** and enabling targeted interventions.

### Ontology-Constrained Dynamics

The dynamics model respects ontology constraints during rollout:

```python
class OntologyConstrainedDynamics(nn.Module):
    """
    Recurrent dynamics model that enforces ontology constraints on latent evolution.
    """
    def step(
        self,
        z_t: StructuredLatentState,
        a_t: OntologyAction | None = None,
    ) -> StructuredLatentState:
        # Standard GRU step
        h_next = self.gru(z_t.to_tensor(), self.h_prev)

        # Decompose into typed partitions
        z_next_raw = self.projection(h_next)
        z_next = StructuredLatentState.from_tensor(z_next_raw, self.ontology_schema)

        # Enforce ontology constraints (e.g., portfolio weights sum to 1)
        z_next = self.ontology_schema.project_to_feasible(z_next)

        # Apply action modulation if provided
        if a_t is not None:
            affected_partitions = self.ontology_schema.get_affected_partitions(a_t)
            for partition in affected_partitions:
                z_next[partition] = z_next[partition] + self.action_heads[partition](a_t)

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
| **Pipeline** | World Model rollout | Latent trajectory `z_t → z_{t+H}` |
| **Contour (view)** | Regime-conditioned projection | Bull-market view of portfolio risk |
| **Workshop (app)** | Agent-facing interface | Portfolio management dashboard |
| **AIP (AI Platform)** | LLM + Ontology integration | Analyst agent querying market context |
| **Branch** | Counterfactual scenario | "What if rates rise 100bps?" simulation |

---

## LLM-Integrated Ontology Queries in Practice

The combination of LLM reasoning with ontology grounding enables queries that neither component could handle alone:

### Example 1 — Contagion Analysis

```python
# Natural language query via LLM agent
query = """
Given the current credit spread widening in European high-yield bonds,
which equity sectors in our portfolio have the highest historical
contagion sensitivity to European credit stress?
"""

# Ontology-grounded response
result = query_engine.query(query)
# Returns: structured graph traversal showing
#   Portfolio → Holdings → Sector exposure
#   Sector → Historical correlation to EU HY credit spread
#   Ranked by contagion beta, with supporting entity references
```

### Example 2 — Regime-Conditional Action Planning

```python
# LLM agent combines ontology context with World Model forecast
def regime_conditional_action(
    agent: LLMAgent,
    world_model: WorldModel,
    query_engine: OntologyQueryEngine,
) -> AgentPlan:
    # Get regime from World Model
    regime_probs = world_model.classify_regime(agent.current_perception.z)

    # Query ontology for regime-specific playbook
    playbook = query_engine.query(
        f"What portfolio adjustments are recommended when "
        f"the market regime is: {regime_probs.most_likely.name}?"
    )

    # Simulate playbook actions
    return agent.simulate_and_rank(playbook.candidate_actions, world_model)
```

### Example 3 — Earnings Event Ontology Update

```python
# When an earnings event occurs, update the ontology and re-run World Model
def process_earnings_event(
    event: EarningsRelease,
    ontology: KnowledgeGraph,
    world_model: WorldModel,
    agent: LLMAgent,
) -> None:
    # Update ontology with new earnings facts
    ontology.update_object(
        class_name='Company',
        object_id=event.company_id,
        updates={
            'last_eps': event.eps_actual,
            'eps_surprise_pct': event.eps_surprise,
            'revenue_actual': event.revenue_actual,
            'guidance_revision': event.guidance,
        }
    )

    # Record the event itself as an ontology object
    ontology.create_object(
        class_name='MarketEvent',
        properties={
            'eventType': 'EarningsRelease',
            'timestamp': event.timestamp,
            'severity': abs(event.eps_surprise),
        },
        relations=[
            ('impacts', event.asset_id, {'weight': event.eps_surprise}),
            ('causedBy', 'EarningsCalendar', {}),
        ]
    )

    # Re-encode market state incorporating the new ontology facts
    updated_perception = agent.perceive(
        raw_observation=agent.latest_observation,
        ontology=ontology,
    )
    # World Model latent state now includes the earnings revision signal
```

---

## Evaluation: Ontology Coverage and LLM Grounding Quality

An ODWM requires evaluation along two new dimensions not present in standard World Models:

### Ontology Coverage Metrics

| Metric | Definition | Target |
|---|---|---|
| **Entity recall** | Fraction of real-world entities present in ontology | > 95% for core assets |
| **Relation precision** | Fraction of stored relations that are factually correct | > 99% |
| **Staleness rate** | Fraction of properties not updated within their expected window | < 1% |
| **Constraint satisfaction** | Fraction of ontology states satisfying declared constraints | 100% |

### LLM Grounding Quality Metrics

| Metric | Definition | Target |
|---|---|---|
| **Hallucination rate** | Fraction of LLM entity references not found in ontology | < 0.5% |
| **Grounding depth** | Average number of ontology hops used in reasoning | ≥ 3 |
| **Answer attribution** | Fraction of LLM answers with traceable ontology evidence | > 90% |
| **Action validity rate** | Fraction of LLM-proposed actions satisfying ontology constraints | > 99% |

---

## Implementation Reference

A minimal ODWM can be constructed by combining three open components:

### 1. Ontology Backend

```python
# Using NetworkX for the knowledge graph (production: RDF triple store or graph DB)
import networkx as nx

class FinancialKnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_entity(self, class_name: str, entity_id: str, properties: dict) -> None:
        self.graph.add_node(entity_id, class_name=class_name, **properties)

    def add_relation(
        self,
        source_id: str,
        relation_type: str,
        target_id: str,
        properties: dict | None = None,
    ) -> None:
        self.graph.add_edge(
            source_id, target_id,
            relation_type=relation_type,
            **(properties or {}),
        )

    def get_neighbours(
        self,
        entity_id: str,
        relation_type: str | None = None,
        max_hops: int = 2,
    ) -> list[dict]:
        result = []
        visited = {entity_id}
        frontier = [entity_id]
        for _ in range(max_hops):
            next_frontier = []
            for node in frontier:
                for _, neighbour, edge_data in self.graph.out_edges(node, data=True):
                    if relation_type is None or edge_data.get('relation_type') == relation_type:
                        if neighbour not in visited:
                            result.append({'id': neighbour, **self.graph.nodes[neighbour]})
                            next_frontier.append(neighbour)
                            visited.add(neighbour)
            frontier = next_frontier
        return result
```

### 2. LLM Agent Adapter

```python
class FinancialLLMAgent:
    """
    Wraps an LLM with ontology-aware tooling for financial reasoning.
    Compatible with OpenAI function calling / Anthropic tool use APIs.
    """
    def __init__(self, llm_client, knowledge_graph: FinancialKnowledgeGraph):
        self.llm = llm_client
        self.kg  = knowledge_graph

        # Expose ontology operations as LLM tools
        self.tools = [
            self._make_entity_lookup_tool(),
            self._make_relation_traversal_tool(),
            self._make_constraint_check_tool(),
        ]

    def reason_with_ontology(self, user_query: str) -> str:
        messages = [{"role": "user", "content": user_query}]
        while True:
            response = self.llm.chat(messages=messages, tools=self.tools)
            if response.stop_reason == "tool_use":
                tool_result = self._dispatch_tool(response.tool_call)
                messages.append({"role": "tool", "content": str(tool_result)})
            else:
                return response.text
```

### 3. World Model Integration

```python
class ODWMFacade:
    """
    Unified interface combining the ontology, LLM agents, and World Model.
    """
    def __init__(
        self,
        world_model: WorldModel,
        knowledge_graph: FinancialKnowledgeGraph,
        llm_agent: FinancialLLMAgent,
    ):
        self.wm    = world_model
        self.kg    = knowledge_graph
        self.agent = llm_agent

    def analyse_and_simulate(
        self,
        question: str,
        current_observation: MarketObservation,
        simulation_horizon: int = 21,
    ) -> AnalysisResult:
        # Step 1: LLM reasons about the question using ontology
        analysis = self.agent.reason_with_ontology(question)

        # Step 2: Extract actionable insight from analysis
        proposed_action = self.agent.extract_action(analysis)

        # Step 3: Encode current state in World Model
        z_t = self.wm.encoder(current_observation)

        # Step 4: Simulate proposed action forward
        latent_paths = self.wm.dynamics.rollout(
            z_0=z_t,
            horizon=simulation_horizon,
            action=proposed_action,
        )
        price_dist = self.wm.decoder(latent_paths)

        return AnalysisResult(
            llm_analysis=analysis,
            proposed_action=proposed_action,
            simulated_price_distribution=price_dist,
        )
```

---

## Chapter Summary

- **Ontologies** provide a formal, machine-readable representation of domain knowledge — entities, relationships, constraints, and action templates — that goes far beyond what a neural network can learn implicitly from data alone
- The **Palantir Foundry paradigm** — objects, semantic layers, action templates, and live updating — translates directly into a powerful architecture for financial World Models
- **LLM-integrated ontologies** enable natural language access to structured knowledge, preventing hallucination by grounding LLM reasoning in verified entities and relationships
- **LLM agents** operating within an Ontology-Driven World Model follow a five-stage loop — Perceive, Reason, Simulate, Act, Observe — each stage enriched by both ontology context and World Model dynamics
- **Multi-agent coordination** through a shared ontology enables specialised agents (Analyst, Portfolio, Risk, Execution, Monitoring) to operate with complementary capabilities while maintaining a consistent knowledge state
- **Ontology-constrained latent spaces** make World Model representations interpretable and ensure that simulated trajectories respect domain constraints
- The combined ODWM architecture represents the state of the art for **explainable, grounded, and agentic financial AI** — moving beyond both pure neural approaches and pure symbolic approaches toward a hybrid system that captures the strengths of each

---

## Looking Ahead

The Ontology-Driven World Model closes the loop on the architecture presented throughout this book. Raw market observations flow through an encoder into a structured latent space, dynamics models simulate forward trajectories, and now LLM agents — grounded in a rich knowledge graph — reason about those trajectories and act upon them with semantic awareness.

This is the architecture of the next generation of financial intelligence: not a single model that predicts prices, but a **cognitive ecosystem** of world models, knowledge graphs, and reasoning agents — each contributing to better decisions under uncertainty.

> *"The ontology is not a constraint on intelligence. It is the scaffolding on which intelligence builds."*
>
> A World Model without structured knowledge can simulate; one with an ontology can **understand**.
