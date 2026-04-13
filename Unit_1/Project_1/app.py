"""
Código para o aplicativo Streamlit do projeto 1 de Deep Learning.
"""

import os
import base64
from PIL import Image
import streamlit as st
from io import BytesIO


def run_app(labels: list[str] = ["cat", "dog", "car", "red"]):
    """
    Dashboard de Alinhamento Visão-Linguagem para LLaVA 1.5

    Este aplicativo Streamlit apresenta uma interface interativa para visualizar o alinhamento entre as representações 
    visuais e textuais no modelo LLaVA 1.5. Ele exibe a imagem de entrada (com corte otimizado para desktop e versão 
    completa para mobile), as distribuições de similaridade de cosseno antes e depois do projetor MLP, e uma 
    projeção t-SNE que mostra como os patches visuais se alinham com a representação textual.
    """
    st.set_page_config(layout="wide", page_title="Dashboard de Alinhamento")

    st.markdown("""
    <style>
    /* Reduce top padding of the main container */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Hide the mobile uncropped image on desktop */
    @media (min-width: 769px) {
        .mobile-img { display: none !important; }
    }

    /* Hide the desktop cropped image and expander on mobile */
    @media (max-width: 768px) {
        .desktop-img { display: none !important; }
        div[data-testid="stExpander"] { display: none !important; }
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Dashboard de Alinhamento Visão-Linguagem (LLaVA 1.5)")
    selected_word = st.selectbox("Selecione um Par Imagem/Palavra:", labels)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(base_dir, "assets")

    def render_image_b64(img_path, class_name):
        if not os.path.exists(img_path):
            return
        img = Image.open(img_path)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        st.markdown(f'<img src="data:image/jpeg;base64,{img_str}" class="{class_name}" style="width:100%; border-radius: 0.25rem;">', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader(f"Imagem de Entrada (Alvo: '{selected_word}')")
        
        render_image_b64(os.path.join(assets_dir, f"{selected_word}_cropped.jpg"), "desktop-img")
        render_image_b64(os.path.join(assets_dir, f"{selected_word}_original.jpg"), "mobile-img")
        
        with st.expander("Clique para ver a imagem COMPLETA e sem cortes"):
            orig_img_path = os.path.join(assets_dir, f"{selected_word}_original.jpg")
            if os.path.exists(orig_img_path):
                st.image(orig_img_path, use_container_width=True)
                st.caption("Esta imagem original é exatamente a que é enviada ao modelo LLaVA para processamento.")
        
        st.subheader("Distribuições de Similaridade (Cosseno)")
        dist_img_path = os.path.join(assets_dir, f"{selected_word}_dist.png")
        if os.path.exists(dist_img_path):
            st.image(dist_img_path, use_container_width=True)
        else:
            st.warning("Gere os gráficos com `python generate_assets.py` primeiro!")

    with col_right:
        st.subheader("Projeção t-SNE: Texto vs Patches Multimodais")
        tsne_img_path = os.path.join(assets_dir, f"{selected_word}_tsne.png")
        if os.path.exists(tsne_img_path):
            st.image(tsne_img_path, use_container_width=True)

if __name__ == "__main__":
    run_app()
