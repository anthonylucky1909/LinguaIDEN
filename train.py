import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.transformer import Transformer
from data.preprocessing import preprocess, train_test_split
from data.dataset import TranslationDataset, collate_fn
from utils.masks import create_src_mask, create_tgt_mask
from utils.helpers import ids_to_text
from config import *
import json
import math

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        src = batch['src'].to(device)
        tgt_in = batch['tgt_in'].to(device)
        tgt_out = batch['tgt_out'].to(device)

        src_mask = create_src_mask(src).to(device)
        tgt_mask = create_tgt_mask(tgt_in).to(device)

        optimizer.zero_grad()
        output = model(src, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask)
        output = output.permute(0, 2, 1)
        loss = criterion(output, tgt_out)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            tgt_in = batch['tgt_in'].to(device)
            tgt_out = batch['tgt_out'].to(device)

            src_mask = create_src_mask(src).to(device)
            tgt_mask = create_tgt_mask(tgt_in).to(device)

            output = model(src, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask)
            output = output.permute(0, 2, 1)
            loss = criterion(output, tgt_out)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Preprocess data
    print("Preprocess data")
    src_ids, tgt_ids_in, tgt_ids_out, src_vocab, tgt_vocab = preprocess(
        'en-id.txt/tico-19.en-id.id', 'en-id.txt/tico-19.en-id.en')
    
    # Save vocabularies
    print("Save vocabularies")
    with open('src_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(src_vocab, f, ensure_ascii=False, indent=2)
    with open('tgt_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(tgt_vocab, f, ensure_ascii=False, indent=2)

    # Split data
    print("Split data")
    (train_src, train_tgt_in, train_tgt_out), (test_src, test_tgt_in, test_tgt_out) = \
        train_test_split(src_ids, tgt_ids_in, tgt_ids_out, TEST_RATIO, SEED)

    # Create datasets
    print("Create datasets")
    train_dataset = TranslationDataset(train_src, train_tgt_in, train_tgt_out)
    test_dataset = TranslationDataset(test_src, test_tgt_in, test_tgt_out)

    # Create dataloaders
    print("Create dataloaders")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, collate_fn=collate_fn)

    # Initialize model
    print("Initialize model")
    model = Transformer(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        input_vocab_size=len(src_vocab),
        target_vocab_size=len(tgt_vocab),
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_val_loss = float('inf')
    print("Train model")
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_transformer_model.pt')
            print(f"Saved best model at epoch {epoch+1}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'transformer_epoch_{epoch+1}.pt')

if __name__ == "__main__":
    main()