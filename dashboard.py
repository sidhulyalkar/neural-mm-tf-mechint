# interpretability/dashboard.py
"""
This module provides a Streamlit dashboard for visualizing attention maps and ablation study results for the multimodal transformer model.

Functions:    
    main: The main function that runs the Streamlit dashboard.
    load_model: Loads the model from a checkpoint.

Imports:    
    streamlit: Module for building Streamlit applications.    
    torch: Module for defining neural network layers.    
    yaml: Module for working with YAML files.    
    model: Module for defining the MultimodalTransformer model.    
    dataloaders: Module for loading data.    
    interpretability.attention_analysis: Module for attention analysis.    
    interpretability.ablation_study: Module for ablation study.    
    interpretability.cav_analysis: Module for concept activation vector analysis.   
"""
import streamlit as st
import torch
import yaml
from model import MultimodalTransformer
from dataloaders import get_loader
from interpretability.attention_analysis import extract_attention_maps
from interpretability.ablation_study import ablate_heads
from interpretability.cav_analysis import compute_cav

@st.cache(allow_output_mutation=True)
def load_model(cfg):
    """
    Loads a MultimodalTransformer model from a checkpoint path specified in the configuration dictionary.

    Args:
        cfg (dict): The configuration dictionary containing the model path.

    Returns:
        model (MultimodalTransformer): The loaded model.
    """
    model = MultimodalTransformer(cfg)
    model.load_state_dict(torch.load(cfg['interpretability']['model_path'], map_location='cpu'))
    model.eval()
    return model

def main():
    """
    The main function that runs the Streamlit dashboard.

    It loads the model, allows the user to specify a fine-tuned checkpoint, and then visualizes the attention maps and ablation study results.

    The dashboard consists of three sections:
    1. Attention visualization: The user can select a layer and head to visualize the attention map.
    2. Ablation study: The user can select a layer and one or more heads to ablate, and then run the ablation study.
    3. Concept activation vector analysis (placeholder): Compute CAVs on hidden activations (experimentally defined concepts).

    """
    st.title("🔍 Multimodal Transformer Interpretability")
    cfg = yaml.safe_load(open('configs/config.yaml'))
    # allow user to specify fine-tuned checkpoint
    cfg['interpretability'] = {'model_path': st.text_input('Model checkpoint path', 'model.pt')}
    model = load_model(cfg)

    # Load a single batch
    loader = get_loader(cfg)
    batch = next(iter(loader))
    inputs = dict(zip(['neural','video','behavior','meta','target'], batch))

    # Attention visualization
    st.header("Attention Maps")
    layer = st.slider('Layer', 0, cfg['model']['num_layers']-1, 0)
    head = st.slider('Head', 0, cfg['model']['n_heads']-1, 0)
    attn = extract_attention_maps(model, inputs, layer, head)
    st.line_chart(attn)

    # Ablation study
    st.header("Ablation Study")
    layer_ab = st.number_input('Layer to ablate', 0, cfg['model']['num_layers']-1, 0)
    heads_to_ablate = st.multiselect('Heads to ablate', list(range(cfg['model']['n_heads'])))
    if st.button('Run Ablation'):
        loss = ablate_heads(model, inputs, layer_ab, heads_to_ablate)
        st.write(f"Post-ablation loss: {loss:.4f}")

    # CAV analysis (placeholder)
    st.header("Concept Activation Vectors")
    st.write("Compute CAVs on hidden activations (experimentally defined concepts).")

if __name__ == '__main__':
    main()