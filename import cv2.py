import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
import os

def ler_imagem(caminho):
    imagem = cv2.imread(caminho)                                                        # Leitura da imagem
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)                                    # Conversão para RGB (se a imagem estiver em BGR)
    return imagem

def obter_quantidade_cores(imagem):
    imagem_convertida = cv2.convertScaleAbs(imagem)                                     # Converter a imagem para um tipo de dados suportado (8 bits por canal)
    imagem_rgb = cv2.cvtColor(imagem_convertida, cv2.COLOR_BGR2RGB)                     # Converter a imagem para o espaço de cores RGB
    pixels = imagem_rgb.reshape((-1, 3))                                                # Redimensionar a imagem para um vetor de pixels
    cores_unicas = np.unique(pixels, axis=0)                                            # Obter cores únicas
    quantidade_cores = len(cores_unicas)                                                # Obter quantidade de cores únicas
    return quantidade_cores

def extrair_propriedades(caminho_imagem, imagem):
    altura, largura, canais = imagem.shape                                              # Obter resolução e canais da imagem
    tamanho_do_arquivo = os.path.getsize(caminho_imagem)                                # Obter tamanho da imagem
    media_cores = np.mean(imagem, axis=(0, 1))                                          # Cálcular média das cores
    quantidade_cores = obter_quantidade_cores(imagem)                                   # Obtem quantidade de cores da imagem
    return {
        "altura": altura,
        "largura": largura,
        "tamanho_do_arquivo (bytes)": tamanho_do_arquivo,
        "quantidade_cores": quantidade_cores,
        "canais": canais,
        "media_cores": media_cores
    }

def aplicar_k_means(imagem, k):
    pixels = imagem.reshape((-1, 3))                                                    # Redimensionar a imagem para um vetor de pixels
    kmeans = KMeans(n_clusters=k)                                                       # Aplicar o algoritmo k-médias
    kmeans.fit(pixels)                                                                  # Encontrar os centros dos clusters
    rótulos = kmeans.labels_                                                            # Obter rótulos dos clusters
    centróides = kmeans.cluster_centers_                                                # Obter centróides dos clusters
    imagem_segmentada = centróides[rótulos].reshape(imagem.shape)                       # Atualizar os pixels da imagem com base nos rótulos dos clusters
    return imagem_segmentada

def plotar_imagens(imagem_segmentada):
    plt.imshow(imagem_segmentada.astype(np.uint8))                                      # Mostrar a imagem usando Matplotlib
    plt.axis('off')                                                                     # Desligar os eixos
    plt.tight_layout()                                                                  # Ajustar o layout para preencher a área disponível
    plt.savefig('image1_3Cores.png', dpi=500, bbox_inches='tight', pad_inches=0.0)      # Exportar a imagem
    """ plt.show()   """                                                                # Exibir a imagem

def main():
    k = 3                                                                                           # Definir a quantidade de cores para a imagem segmentada
    caminho_imagem_original = "1.png"                                                               # Definir caminho da imagem original
    imagem_original = ler_imagem(caminho_imagem_original)                                           # Realizar a leitura da imagem original

    propriedades_originais = extrair_propriedades(caminho_imagem_original, imagem_original)         # Extrair as propriedades da imagem original e imprimir
    print("Propriedades da Imagem Original:")
    print(propriedades_originais) 

    imagem_segmentada = aplicar_k_means(imagem_original, k)                                         # Aplicar o K-médias a imagem original                          

    plotar_imagens(imagem_segmentada)                                                               # Exportar e visualizar a imagem segmentada
    
    caminho_imagem_segmentada = "image1_3Cores.png"                                                 # Definir caminho da imagem segmentada
    propriedades_segmentadas = extrair_propriedades(caminho_imagem_segmentada, imagem_segmentada)   # Extrair as propriedades da imagem segmentada e imprimir
    print(f"\nPropriedades da Imagem Segmentada (k={k}):")
    print(propriedades_segmentadas)


if __name__ == "__main__":
    main()