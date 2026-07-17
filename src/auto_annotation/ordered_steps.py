from auto_annotation.domain.models import OrderedTask, TaskStep
from auto_annotation.ontology.models import OntologyEntry
from auto_annotation.ontology.registry import ResolvedContract


def validate_ordered_task_contract(
    task: OrderedTask,
    contract: ResolvedContract,
) -> tuple[OntologyEntry, ...]:
    if task.schema_id != contract.schema.schema_id:
        raise ValueError(
            "ordered task schema mismatch: "
            f"expected {task.schema_id}, loaded {contract.schema.schema_id}"
        )
    if task.ontology_id != contract.ontology.ontology_id:
        raise ValueError(
            "ordered task ontology mismatch: "
            f"expected {task.ontology_id}, loaded {contract.ontology.ontology_id}"
        )
    return resolve_ordered_step_entries(task.steps, contract)


def resolve_ordered_step_entries(
    steps: tuple[TaskStep, ...],
    contract: ResolvedContract,
) -> tuple[OntologyEntry, ...]:
    if len(steps) < 2:
        raise ValueError(
            "ordered-step full-coverage annotation requires at least two steps"
        )
    step_ids = [step.step_id for step in steps]
    if len(set(step_ids)) != len(step_ids):
        raise ValueError("ordered-step IDs must be unique")

    entries = tuple(
        contract.entry_for("ontology.atomic_actions", step.step_id)
        for step in steps
    )
    for step, entry in zip(steps, entries, strict=True):
        contract.validate_values(step.values)
        if step.values.get("atomic_action") != step.step_id:
            raise ValueError(
                "ordered step atomic_action must match step_id: "
                f"{step.step_id}"
            )
        if entry.temporal_type is None or entry.boundary_rule is None:
            raise ValueError(
                "ordered-step ontology entry requires temporal metadata: "
                f"{entry.id}"
            )
    return entries
