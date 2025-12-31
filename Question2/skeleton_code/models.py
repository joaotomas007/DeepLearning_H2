import torch
import torch.nn as nn
import torch.nn.functional as F

class RBFOX1_CNN(nn.Module):
    def __init__(self, num_filters=32, kernel_size=12):
        # 1. Herança (Boilerplate obrigatório)
        super().__init__() 
        
        # 2. Definir as "Peças" (Camadas)
        
        # O "Scanner": Conv1d
        # in_channels=4: Porque o RNA tem 4 letras (A, C, G, U)
        # out_channels=num_filters: Quantos padrões diferentes queremos procurar? (ex: 32)
        # kernel_size: Tamanho da janela do scanner
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=num_filters, kernel_size=kernel_size)

        # O "Interruptor": ReLU
        # Transforma valores negativos em zero (ativação)
        self.relu = nn.ReLU()
        
        # O "Resumidor": MaxPool
        # Olha para toda a sequência e guarda apenas o valor mais alto encontrado
        # O '1' significa que no final queremos apenas 1 valor por filtro
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Recebe 32, cospe 16 (reduz a dimensão para "pensar" melhor)
        self.fc1 = nn.Linear(num_filters, 16)

        # O "Classificador": Linear
        # A saída final recebe 16 e cospe 1
        self.fc2 = nn.Linear(in_features=16, out_features=1)

    def forward(self, x):
        # --- PASSO 0: A Entrada ---
        # O x chega com o formato: [Batch_Size, 41, 4]
        # Exemplo: [64 sequências, 41 letras de comprimento, 4 canais A/C/G/U]
        
        # --- PASSO 1: A Rotação (Permute) ---
        # Problema: A camada Conv1d do PyTorch espera receber [Batch, Canais, Comprimento].
        # Solução: Temos de trocar a dimensão 1 (41) com a dimensão 2 (4).
        x = x.permute(0, 2, 1)
        # Agora o x está: [Batch_Size, 4, 41]. Pronto para entrar no scanner!

        # --- PASSO 2: A Convolução + ReLU ---
        # Passamos pelos 32 filtros e aplicamos a ReLU para eliminar negativos.
        x = self.conv1(x)
        x = self.relu(x)
        # O x agora é um mapa de características.
        # Formato: [Batch_Size, 32, 30] 
        # (32 filtros diferentes. O comprimento encolheu de 41 para 30 por causa do tamanho do filtro 12)

        # --- PASSO 3: O Resumo (Max Pooling) ---
        # Perguntamos: "Qual foi o valor máximo que cada filtro encontrou?"
        x = self.pool(x)
        # Formato: [Batch_Size, 32, 1]
        # (Ainda temos 3 dimensões, mas a última é inútil porque é apenas tamanho 1)

        # --- PASSO 4: A Limpeza (Squeeze) ---
        # A camada Linear não gosta de caixas 3D. Ela quer uma lista plana de números.
        # O squeeze remove dimensões de tamanho 1.
        x = x.squeeze(-1) 
        # Formato: [Batch_Size, 32]
        # (Agora sim: para cada uma das 64 sequências, temos 32 números de resumo)

        # --- PASSO 5: A Decisão Final (Linear) ---
        # O Juiz dá a nota final baseada nos 32 resumos.
        x = self.fc1(x)       # Transforma 32 em 16
        x = self.relu(x)      # Corta os negativos
        
        # 6. Segunda Camada Linear (Saída)
        x = self.fc2(x)       # Transforma 16 em 1 (Score Final)
        # Formato: [Batch_Size, 1]
        # (Um único valor de afinidade para cada sequência)

        return x
    
class RBFOX1_LSTM(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        # 1. A Camada LSTM
        # input_size=4: Entram 4 letras (A, C, G, U) de cada vez.
        # hidden_size=32: A "capacidade de memória" da rede. Quanto maior, mais complexa.
        # batch_first=True: Diz ao PyTorch que os nossos dados vêm como [Batch, Comprimento, Canais]
        # bidirectional=True: Cria duas LSTMs (uma lê normal, outra lê ao contrário)
        self.lstm = nn.LSTM(input_size=4, 
                            hidden_size=hidden_dim, 
                            batch_first=True, 
                            bidirectional=True)
        
        # 2. A Camada Linear Final
        # ATENÇÃO À MATEMÁTICA AQUI:
        # Se a LSTM é bidirecional, ela cospe 32 valores da ida + 32 valores da volta.
        # Logo, a entrada da camada Linear é hidden_dim * 2.
        self.fc = nn.Linear(in_features=hidden_dim * 2, out_features=1)

    def forward(self, x):
        # Input: [Batch, 41, 4] -> A LSTM come isto diretamente (não precisa de permute!)
        
        # A LSTM devolve: output, (hidden_state, cell_state)
        output, (hn, cn) = self.lstm(x)
        
        # Truque Ninja: Vamos buscar o estado final da ida (hn[-2]) e da volta (hn[-1])
        # e colamo-los um ao outro (cat).
        final_state = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        
        x = self.fc(final_state)
        return x

class RBFOX1_LSTM_Attention(nn.Module):
    def __init__(self, hidden_dim=64): # Vamos manter os 64 que deram bom resultado
        super().__init__()
        
        # 1. A mesma LSTM de antes
        self.lstm = nn.LSTM(input_size=4, 
                            hidden_size=hidden_dim, 
                            batch_first=True, 
                            bidirectional=True)
        
        # 2. Camada de Atenção
        # Esta camada vai olhar para o estado escondido de cada nucleótido
        # e decidir quão importante ele é (devolve 1 valor de score)
        self.attention_linear = nn.Linear(hidden_dim * 2, 1)
        
        # 3. Classificador Final
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x shape: [Batch, 41, 4]
        
        # 1. Passar pela LSTM
        # lstm_out shape: [Batch, 41, hidden_dim * 2]
        # Aqui temos a "memória" de cada posição, não só a final!
        lstm_out, _ = self.lstm(x)
        
        # 2. Calcular Scores de Atenção (Energy)
        # Transformamos [Batch, 41, Hidden*2] -> [Batch, 41, 1]
        attn_scores = self.attention_linear(lstm_out)
        
        # 3. Transformar em Probabilidades (Softmax)
        # Queremos que a soma das importâncias seja 100% (1.0)
        attn_weights = F.softmax(attn_scores, dim=1) 
        
        # 4. Média Ponderada (Context Vector)
        # Multiplicamos cada estado da LSTM pelo seu peso e somamos tudo
        # [Batch, 41, Hidden*2] * [Batch, 41, 1] -> Soma -> [Batch, Hidden*2]
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        
        # 5. Previsão Final
        out = self.fc(context_vector)
        
        return out