// Código adaptado de: https://www.geeksforgeeks.org/dsa/strassens-matrix-multiplication/

// [Approach 1] - Using Nested Loops - O(n * m * q) Time and O(n * q) Space

// The main idea is to initializes the result matrix with zeros and fills each cell
// by computing the dot product of the corresponding row from a and column from .
// This is done using three nested loops: the outer two iterate over the result matrix cells,
// and the innermost loop performs the multiplication and addition.

#include <vector>
using namespace std;

vector<vector<int>> multiplyNaive(vector<vector<int>> &mat1, 
                                        vector<vector<int>> &mat2) {
    
    int n = mat1.size(), m = mat1[0].size(), 
                                    q = mat2[0].size();        

    // Initialize the result matrix with 
    // dimensions n×q, filled with 0s
    vector<vector<int>> res(n, vector<int>(q, 0));

    // Loop through each row of mat1
    for (int i = 0; i < n; i++) {
        
        // Loop through each column of mat2
        for (int j = 0; j < q; j++) {
            
            // Compute the dot product of 
            // row mat1[i] and column mat2[][j]
            for (int k = 0; k < m; k++) {
                res[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }

    return res;
}