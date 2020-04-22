

def get_matrix_str(mat):
    m, n = mat.shape
    s = '\n\t'

    for i in range(n):
        s += f'  {i}\t'
    s += '\n\n'

    for i in range(m):
        s += f'{i}\t'
        for j in range(n):
            num = mat[i, j]
            s += ' {:.2f}\t'.format(num) if num >= 0 else '{:.2f}\t'.format(num)
        s += '\n'

    return s
