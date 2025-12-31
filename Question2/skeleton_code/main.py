import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from config import RNAConfig
from utils import load_rnacompete_data, configure_seed, masked_mse_loss, masked_spearman_correlation, plot
from models import RBFOX1_CNN, RBFOX1_LSTM, RBFOX1_LSTM_Attention

def train_model(model_type='CNN'):
    # --- 1. SETUP INICIAL ---
    config = RNAConfig()
    configure_seed(config.SEED)
    
    # Detetar se temos GPU (Cuda) ou CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--> A usar o dispositivo: {device}")
    
    # --- 2. PREPARAR DADOS ---
    print("--> A carregar dados...")
    # O dataset contém TUDO
    train_dataset = load_rnacompete_data('RBFOX1', split='train', config=config)
    val_dataset = load_rnacompete_data('RBFOX1', split='val', config=config)
    
    # O DataLoader divide em 'Batches' (Pacotes de 64 sequências)
    # Shuffle=True no treino para baralhar e o modelo não decorar a ordem
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # --- 3. INICIALIZAR MODELO ---
    if model_type == 'CNN':
        print("--> A inicializar CNN...")
        model = RBFOX1_CNN(num_filters=32, kernel_size=12)
    elif model_type == 'LSTM':
        print("--> A inicializar LSTM...")
        model = RBFOX1_LSTM(hidden_dim=64)
    elif model_type == 'LSTM_Attn':
        print("--> A carregar modelo LSTM + Attention...")
        # Usamos os mesmos 64 para ser uma comparação justa!
        model = RBFOX1_LSTM_Attention(hidden_dim=64)
    
    # Mover o modelo para a GPU/CPU
    model = model.to(device)
    
    # --- 4. CONFIGURAR TREINO ---
    # Otimizador Adam: O mecânico que ajusta os pesos
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # lr = Learning Rate (Velocidade de aprendizagem)
    epochs = 50 # Quantas vezes vamos ver o dataset inteiro?
    
    # Listas para guardar o histórico e fazer gráficos no fim
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_spearman': []
    }

    # --- 5. LOOP DE TREINO (A Ação!) ---
    print("--> A iniciar treino...")
    
    for epoch in range(epochs):
        
        # === FASE DE TREINO ===
        model.train() # Coloca o modelo em modo "Aluno" (ativa Dropout, etc)
        epoch_train_loss = 0.0
        
        for batch in train_loader:
            # Desempacotar e enviar para GPU
            x, y, mask = batch
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            # 1. Forward (Previsão)
            preds = model(x)
            
            # 2. Calcular Erro (Loss)
            loss = masked_mse_loss(preds, y, mask)
            
            # 3. Backward (Calcular correções)
            optimizer.zero_grad() # Limpar correções antigas
            loss.backward()
            
            # 4. Step (Aplicar correções)
            optimizer.step()
            
            # Acumular o erro para a média da época
            epoch_train_loss += loss.item()
            
        # Média do erro de treino
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        # === FASE DE VALIDAÇÃO ===
        model.eval() # Coloca o modelo em modo "Exame" (congela pesos)
        epoch_val_loss = 0.0
        epoch_val_spearman = 0.0
        
        with torch.no_grad(): # Desliga o cálculo de gradientes (poupa memória)
            for batch in val_loader:
                x, y, mask = batch
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                
                preds = model(x)
                
                # Calcular métricas
                loss = masked_mse_loss(preds, y, mask)
                spearman = masked_spearman_correlation(preds, y, mask)
                
                epoch_val_loss += loss.item()
                epoch_val_spearman += spearman.item()
        
        # Médias de validação
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_spearman = epoch_val_spearman / len(val_loader)
        
        # Guardar histórico
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_spearman'].append(avg_val_spearman)
        
        # Print de progresso
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Spearman: {avg_val_spearman:.4f}")

    print("--> Treino concluído!")
    
    # --- 6. GRÁFICOS ---
    # Usar a função plot do utils.py
    # Gráfico 1: Loss (Erro) - Deve descer
    plot(range(epochs), {
        'Train Loss': history['train_loss'], 
        'Val Loss': history['val_loss']
    }, filename=f"loss_curve_{model_type}.png")
    
    # Gráfico 2: Spearman (Qualidade) - Deve subir
    plot(range(epochs), {
        'Val Spearman': history['val_spearman']
    }, filename=f"spearman_curve_{model_type}.png")
    
    print(f"Gráficos guardados como loss_curve_{model_type}.png e spearman_curve_{model_type}.png")

if __name__ == "__main__":
    # Experimenta correr a CNN primeiro
    train_model(model_type='LSTM_Attn')