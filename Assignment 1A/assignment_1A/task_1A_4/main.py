import gmm_em

if __name__ == '__main__':
    x_cords = []
    y_cords = []
    magnitude = []
    with open('./data/X.txt', 'r') as file:
        text = file.read()
        rows = text.split('\n')
        for row in rows:
            x, y = row.split(' ')
            x_cords.append(float(x))
            y_cords.append(float(y))
    with open('./data/S.txt', 'r') as file:
        text = file.read()
        rows = text.split('\n')
        for row in rows:
            magnitude.append(int(float(row)))

    gmm_em.gmm_em(x_cords, y_cords, magnitude, k=3)