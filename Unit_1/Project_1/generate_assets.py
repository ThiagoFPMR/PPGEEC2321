import os
import torch
import requests
import seaborn as sns
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from transformers import LlavaProcessor, LlavaForConditionalGeneration


def load_llava():
    model_id = "llava-hf/llava-1.5-7b-hf"

    processor = LlavaProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        device_map="auto",
        low_cpu_mem_usage=True
    )

    return processor, model

def get_img_from_url(img_url, save_path=None):
    """
    Downloads the image from the URL and returns both the original version and a cropped version for better dashboard visualization.

    Args:
        img_url (str): URL of the image to download.
        save_path (str, optional): If provided, saves the original and cropped images to disk with this base path. 
                                   Must include base filename without extension (e.g., "assets/cat").
    """
    original_img = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    
    target_width, target_height = original_img.width, int(original_img.width / 2.5)
    cropped_img = ImageOps.fit(original_img, (target_width, target_height), Image.Resampling.LANCZOS)

    if save_path:
        original_img.save(f"{save_path}_original.jpg")
        cropped_img.save(f"{save_path}_cropped.jpg")

    return original_img

def distribution_plot(sim_before, sim_after, save_path):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    sns.kdeplot(sim_before, fill=True, label='CLIP Bruto (Preenchido)', color='blue', alpha=0.6, ax=ax)
    sns.kdeplot(sim_after, fill=True, label='Projetado (MLP)', color='green', alpha=0.6, ax=ax)
    ax.set_title('Densidade de Similaridade de Cosseno vs Texto')
    ax.set_xlabel('Similaridade de Cosseno')
    ax.set_ylabel('Densidade')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def tsne_plot(all_embs, raw_visual_padded, save_path):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, metric='cosine')
    coords = tsne.fit_transform(all_embs)

    fig_tsne, ax_tsne = plt.subplots(figsize=(10, 6.2))
    ax_tsne.scatter(coords[0, 0], coords[0, 1], marker='x', color='red', s=250, zorder=5, label='Texto')
    c1 = 1
    c2 = 1 + len(raw_visual_padded)
    ax_tsne.scatter(coords[c1:c2, 0], coords[c1:c2, 1], marker='o', color='blue', s=40, alpha=0.5, label='Patches CLIP Brutos')
    ax_tsne.scatter(coords[c2:, 0], coords[c2:, 1], marker='v', color='green', s=40, alpha=0.5, label='Patches Projetados (MLP)')
    ax_tsne.set_title("Projeção t-SNE: Texto vs Patches Multimodais")
    ax_tsne.legend()
    ax_tsne.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def np_consine_similarity(a, b):
    return cosine_similarity(a.cpu().float().numpy(), b.cpu().float().numpy())

def generate_assets(data_samples, out_dir="assets"):
    """
    Generates and saves the necessary images and plots for the Streamlit dashboard. This includes:
    - Downloading the original images from the specified URLs.
    - Creating cropped versions of the images for better visualization on desktop.
    - Computing the cosine similarity distributions before and after the MLP projector.
    - Creating t-SNE projections of the text and visual embeddings.
    - Saving all generated assets (images and plots) to the specified output directory.

    Args:
        data_samples (list): List of dictionaries containing the word and image URL for each sample.
        out_dir (str): Directory where the generated assets will be saved. Defaults to "assets".
    """
    os.makedirs(out_dir, exist_ok=True)
    processor, model = load_llava()

    for sample in data_samples:
        word = sample["word"]
        url = sample["url"]
        print(f"Generating assets for '{word}'...")
        image = get_img_from_url(url, f"{out_dir}/{word}")
        
        text_inputs = processor.tokenizer(word, return_tensors="pt").to("cuda")
        with torch.no_grad():
            text_embedding = model.get_input_embeddings()(text_inputs.input_ids).mean(dim=1)
            
        image_inputs = processor.image_processor(images=image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            vision_outputs = model.model.vision_tower(image_inputs.pixel_values, output_hidden_states=True)
            raw_visual_embeddings = vision_outputs.last_hidden_state[0]
            projected_visual_embeddings = model.model.multi_modal_projector(raw_visual_embeddings)      
        raw_visual_padded = torch.nn.functional.pad(raw_visual_embeddings, (0, 4096 - raw_visual_embeddings.shape[1]))
        
        sim_before_array = np_consine_similarity(raw_visual_padded, text_embedding).flatten()
        sim_after_array = np_consine_similarity(projected_visual_embeddings, text_embedding).flatten()
        
        all_embs = torch.cat([text_embedding, raw_visual_padded, projected_visual_embeddings], dim=0).cpu().float().numpy()
        
        distribution_plot(sim_before_array, sim_after_array, f"{out_dir}/{word}_dist.png")   
        tsne_plot(all_embs, raw_visual_padded, f"{out_dir}/{word}_tsne.png")

    print("Done! Images and plots saved in the 'assets/' folder. Run 'streamlit run app.py' to view.")


if __name__ == "__main__":
    data_samples = [
        {"word": "cat", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
        {"word": "dog", "url": "https://images.unsplash.com/photo-1517849845537-4d257902454a"},
        {"word": "car", "url": "https://images.unsplash.com/photo-1494976388531-d1058494cdd8"},
        {"word": "red", "url": "https://dummyimage.com/1024x1024/ff0000/ff0000.png"}
    ]
    generate_assets(data_samples)
