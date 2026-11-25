"""Expression IG and optimized edge masks for frozen GATv2/GINE predictions."""
import numpy as np
import torch
from captum.attr import IntegratedGradients
from torch_geometric.explain import Explainer, GNNExplainer


def margin(model, graph, expression=None):
    batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=graph.x.device)
    with torch.no_grad():
        logits = model(graph.x if expression is None else expression,
                       graph.edge_index, graph.edge_attr, batch, graph.gene_id)
    return float(logits[0, 1] - logits[0, 0])


def integrated_gradients(model, graph, baseline, steps, internal_batch):
    """Replicate the fixed graph for Captum's interpolation samples."""
    n = graph.num_nodes
    edges = graph.edge_index.shape[1]
    def forward(inputs):
        count = len(inputs)
        offsets = torch.arange(count, device=inputs.device)
        edge_index = (graph.edge_index[None] + offsets[:, None, None] * n).permute(1, 0, 2).reshape(2, count * edges)
        logits = model(inputs.reshape(count * n, 1), edge_index,
                       graph.edge_attr.repeat(count, 1), offsets.repeat_interleave(n),
                       graph.gene_id.repeat(count))
        return logits[:, 1] - logits[:, 0]
    values, delta = IntegratedGradients(forward).attribute(
        graph.x[None], baselines=baseline[None], method="riemann_middle",
        n_steps=steps, internal_batch_size=internal_batch, return_convergence_delta=True)
    return values[0, :, 0].detach().cpu().numpy(), float(delta[0])


def adaptive_ig(model, graph, baseline, initial, maximum, internal, tolerance):
    original = margin(model, graph)
    reference = margin(model, graph, baseline)
    steps = initial
    while True:
        values, delta = integrated_gradients(model, graph, baseline, steps, internal)
        error = abs(delta) / (1 + abs(original - reference))
        if error <= tolerance or steps >= maximum:
            break
        steps = min(steps * 2, maximum)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite IG values")
    if not np.isclose(values.sum(), original - reference + delta, atol=1e-3, rtol=1e-4):
        raise ValueError("IG completeness bookkeeping mismatch")
    return values, dict(margin=original, baseline_margin=reference, captum_delta=delta,
                        relative_completeness_error=error, converged=error <= tolerance, steps=steps)


def edge_explanation(model, graph, epochs):
    """Explain the predicted class; edge masks are conditional computational evidence."""
    original = margin(model, graph)
    target = int(original >= 0)
    batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=graph.x.device)
    explainer = Explainer(model=model, algorithm=GNNExplainer(epochs=epochs, lr=.01,
                         edge_size=.0001, node_feat_size=.01), explanation_type="phenomenon",
                         node_mask_type="attributes", edge_mask_type="object",
                         model_config=dict(mode="multiclass_classification", task_level="graph", return_type="raw"))
    explanation = explainer(graph.x, graph.edge_index, target=torch.tensor([target], device=graph.x.device),
                            edge_attr=graph.edge_attr, batch=batch, gene_id=graph.gene_id)
    if not np.isclose(margin(model, graph), original, atol=1e-5 * (1 + abs(original)), rtol=0):
        raise ValueError("GNNExplainer did not restore the unmasked prediction")
    return explanation.edge_mask.detach().cpu().numpy(), explanation.node_mask[:, 0].detach().cpu().numpy(), target
