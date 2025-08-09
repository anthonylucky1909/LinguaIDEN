import torch
import json
from models.transformer import Transformer
from utils.helpers import ids_to_text
from config import *

def translate_id_to_en(sentence_id, model, src_vocab, tgt_vocab, id_to_word, device, max_length=50):
    tokens = sentence_id.lower().split()
    src_ids = [src_vocab.get(token, src_vocab['<unk>']) for token in tokens]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)

    tgt_input = torch.tensor([[tgt_vocab['<sos>']]], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_length):
            src_mask = (src != src_vocab['<pad>']).unsqueeze(1).unsqueeze(2).to(device)
            tgt_mask = torch.tril(torch.ones((tgt_input.size(1), tgt_input.size(1)), device=device)).bool()
            tgt_mask = tgt_mask.unsqueeze(0)

            output = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
            next_token = output.argmax(dim=-1)[:, -1].unsqueeze(1)
            tgt_input = torch.cat([tgt_input, next_token], dim=-1)

            if next_token.item() == tgt_vocab['<eos>']:
                break

    translation = []
    for id in tgt_input[0].cpu().numpy():
        if id == tgt_vocab['<eos>']:
            break
        if id not in [tgt_vocab['<pad>'], tgt_vocab['<sos>']]:
            translation.append(id_to_word[id])

    return ' '.join(translation)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocabularies
    with open('src_vocab.json', 'r', encoding='utf-8') as f:
        src_vocab = json.load(f)
    with open('tgt_vocab.json', 'r', encoding='utf-8') as f:
        tgt_vocab = json.load(f)
    id_to_word = {v: k for k, v in tgt_vocab.items()}

    # Initialize model
    model = Transformer(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        input_vocab_size=len(src_vocab),
        target_vocab_size=len(tgt_vocab),
        max_seq_len=MAX_SEQ_LEN
    ).to(device)

    # Load trained weights
    model.load_state_dict(torch.load('best_transformer_model.pt', map_location=device))
    model.eval()

    # Example translation
    input_id = "berdasarkan hal ini, gilead menyediakan senyawa ini bagi tiongkok untuk melakukan serangkaian uji coba pada orang terinfeksi sars-cov-2, dan hasilnya sangat dinantikan."
    output_en = translate_id_to_en(input_id, model, src_vocab, tgt_vocab, id_to_word, device)
    print(f"Indonesian: {input_id}")
    print(f"English: {output_en}")

if __name__ == "__main__":
    main()