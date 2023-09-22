import cv2
import numpy as np
import time  # Importe o módulo time

# Carregue imagens de referência para os produtos
ts3110_image = cv2.imread('TS3110.jpg', cv2.IMREAD_GRAYSCALE)
ls1005g_image = cv2.imread('LS1005.jpg', cv2.IMREAD_GRAYSCALE)
fx2100_image = cv2.imread('FX2100.jpg', cv2.IMREAD_GRAYSCALE)

# Inicialize o detector ORB
orb = cv2.ORB_create()

# Extraia os pontos de interesse e descritores para cada imagem de referência
keypoints_ts3110, descriptors_ts3110 = orb.detectAndCompute(ts3110_image, None)
keypoints_ls1005g, descriptors_ls1005g = orb.detectAndCompute(ls1005g_image, None)
keypoints_fx2100, descriptors_fx2100 = orb.detectAndCompute(fx2100_image, None)

# Armazene as imagens de referência e descritores em listas
reference_images = [ts3110_image, ls1005g_image, fx2100_image]
reference_descriptors = [descriptors_ts3110, descriptors_ls1005g, descriptors_fx2100]

# Crie rótulos para os produtos correspondentes
product_labels = ["TS3110", "LS1005G", "FX2100"]

# Inicialize a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converta o quadro da webcam para escala de cinza
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Encontre pontos de interesse e descritores no quadro da webcam
    orb = cv2.ORB_create()
    keypoints_frame, descriptors_frame = orb.detectAndCompute(gray_frame, None)

    # Verifique se os descritores de referência e do quadro da webcam são válidos
    if descriptors_frame is not None:
        # Compare os descritores do quadro da webcam com os descritores de todas as imagens de referência
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        best_match = None
        best_match_index = -1

        for i, reference_descriptor in enumerate(reference_descriptors):
            if reference_descriptor is None:
                continue

            # Verifique o tipo de dados dos descritores
            if reference_descriptor.dtype != descriptors_frame.dtype:
                continue

            # Verifique as dimensões dos descritores
            if reference_descriptor.shape[1] != descriptors_frame.shape[1]:
                continue

            matches = bf.match(reference_descriptor, descriptors_frame)
            matches = sorted(matches, key=lambda x: x.distance)
            match_threshold = 50

            valid_matches = [match for match in matches if match.distance < match_threshold]

            if len(valid_matches) > 10:  # Ajuste conforme necessário
                if best_match is None or len(valid_matches) > len(best_match):
                    best_match = valid_matches
                    best_match_index = i

        # Desenhe uma caixa delimitadora e rótulo se um produto for detectado
        # Desenhe uma caixa delimitadora e rótulo se um produto for detectado
            # Desenhe uma caixa delimitadora e rótulo se um produto for detectado
            # Desenhe uma caixa delimitadora e rótulo se um produto for detectado
            # Desenhe uma caixa delimitadora e rótulo se um produto for detectado
# Desenhe uma caixa delimitadora e rótulo se um produto for detectado
            if best_match_index >= 0:
                img = reference_images[best_match_index]
                h, w = img.shape
                x, y = keypoints_frame[0].pt
                
                # Defina as proporções para redimensionar a caixa
                scale_factor = 0.2 # Ajuste o fator de escala conforme necessário (50% no exemplo)
                
                # Calcule as novas coordenadas da caixa delimitadora
                box_x1, box_y1 = int(x - w * scale_factor / 2), int(y - h * scale_factor / 2)
                box_x2, box_y2 = int(x + w * scale_factor / 2), int(y + h * scale_factor / 2)
                
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
                
                # Escreva o nome do produto acima da caixa
                text = product_labels[best_match_index]
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x = int((box_x1 + box_x2 - text_size[0]) / 2)
                text_y = box_y1 - 10  # Coloque o texto 10 pixels acima da caixa
                
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)





    
    # Exiba o quadro com a caixa delimitadora (se aplicável)
    cv2.imshow('Detecção de Produtos', frame)
    time.sleep(1.0)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere a captura e feche a janela
cap.release()
cv2.destroyAllWindows()
