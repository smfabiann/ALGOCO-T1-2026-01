#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include "algorithms/naive.cpp"
#include "algorithms/strassen.cpp"
using namespace std;
using namespace std::chrono;

long get_status_memory_kb(const string& field_name) {
    ifstream status("/proc/self/status");
    string line;
    while (getline(status, line)) {
        if (line.rfind(field_name, 0) == 0) {
            size_t first_digit = line.find_first_of("0123456789");
            if (first_digit == string::npos) return -1;
            size_t end_digit = line.find_first_not_of("0123456789", first_digit);
            return stol(line.substr(first_digit, end_digit - first_digit));
        }
    }
    return -1;
}

long get_current_rss_kb() {
    return get_status_memory_kb("VmRSS:");
}

long get_peak_rss_kb() {
    return get_status_memory_kb("VmHWM:");
}

struct IterationMetrics {
    double time_ms;
    long rss_before_kb;
    long rss_after_kb;
    long delta_rss_kb;
    long peak_kb;
    long delta_peak_from_start_kb;
};

// Guarda la matriz en un archivo
void guardar_matriz(const vector<vector<int>>& mat, const string& outfilename) {
    ofstream outmat("data/matrix_output/" + outfilename);
    for (const auto& row : mat) {
        for (size_t j = 0; j < row.size(); ++j) {
            outmat << row[j] << (j + 1 < row.size() ? " " : "");
        }
        outmat << "\n";
    }
    outmat.close();
}

bool check_completed(const string& algo) {
    ifstream infile("data/measurements/completion_list.txt");
    string linea;
    while (getline(infile, linea)) {
        if (linea == algo) return true;
    }
    return false;
}

void add_completed(const string& algo) {
    ofstream outfile("data/measurements/completion_list.txt", ios::app);
    outfile << algo << "\n";
    outfile.close();
}

vector<vector<int>> leer_matriz_txt(const string& filename, int n) {
    ifstream infile(filename);
    if (!infile) {
        cerr << "No se pudo abrir el archivo: " << filename << endl;
        return {};
    }
    vector<vector<int>> mat(n, vector<int>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            infile >> mat[i][j];
    infile.close();
    return mat;
}

IterationMetrics run_iteration_isolated(
    const string& algo,
    int n,
    const string& t,
    const string& d,
    const string& m,
    vector<vector<int>>& mA,
    vector<vector<int>>& mB
) {
    IterationMetrics metrics{};
    int pipefd[2];
    if (pipe(pipefd) == -1) {
        cerr << "Error al crear pipe para la medicion." << endl;
        return metrics;
    }

    pid_t pid = fork();
    if (pid < 0) {
        cerr << "Error al crear proceso hijo para la medicion." << endl;
        close(pipefd[0]);
        close(pipefd[1]);
        return metrics;
    }

    if (pid == 0) {
        close(pipefd[0]);

        vector<vector<int>> resultado;
        long rss_before = get_current_rss_kb();

        auto start = high_resolution_clock::now();
        if (algo == "strassen") {
            resultado = multiplyStrassen(mA, mB);
        } else {
            resultado = multiplyNaive(mA, mB);
        }
        auto end = high_resolution_clock::now();

        long rss_after = get_current_rss_kb();
        struct rusage usage {};
        getrusage(RUSAGE_SELF, &usage);
        long peak_kb = usage.ru_maxrss;

        IterationMetrics child_metrics;
        child_metrics.time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        child_metrics.rss_before_kb = rss_before;
        child_metrics.rss_after_kb = rss_after;
        child_metrics.delta_rss_kb =
            (rss_before >= 0 && rss_after >= 0) ? (rss_after - rss_before) : -1;
        child_metrics.peak_kb = peak_kb;
        child_metrics.delta_peak_from_start_kb =
            (rss_before >= 0 && peak_kb >= 0) ? (peak_kb - rss_before) : -1;

        guardar_matriz(
            resultado,
            algo + "_" + to_string(n) + "_" + t + "_" + d + "_" + m + "_out.txt"
        );

        ssize_t written = write(pipefd[1], &child_metrics, sizeof(child_metrics));
        close(pipefd[1]);

        if (written != sizeof(child_metrics)) {
            _exit(2);
        }
        _exit(0);
    }

    close(pipefd[1]);
    ssize_t read_bytes = read(pipefd[0], &metrics, sizeof(metrics));
    close(pipefd[0]);

    int status = 0;
    waitpid(pid, &status, 0);

    if (read_bytes != sizeof(metrics) || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        cerr << "Fallo al medir la iteracion en proceso aislado." << endl;
        metrics = {};
    }

    return metrics;
}

int main() {
    // Usamos tamaños, tipos, dominios y muestras para construir los nombres de archivo
    vector<int> ene = {16, 64, 256, 1024};
    vector<string> tipo = {"dispersa", "diagonal", "densa"};
    vector<string> dominio = {"D0", "D10"};
    vector<string> muestra = {"a", "b", "c"};

    // Algoritmos a medir
    vector<string> algoritmo = {"strassen", "naive"};

    for (string algo : algoritmo) {
        // si el algoritmo se completo, saltalo
        if (check_completed(algo)) continue;

        cout << "----------------------> Ejecutando algoritmo: " << algo << " <----------------------\n";

        ofstream outfile("data/measurements/" + algo + ".csv");
        outfile << "size,tipo,dominio,muestra,time_ms,rss_before_kb,rss_after_kb,delta_rss_kb,peak_kb,delta_peak_from_start_kb\n";
        outfile.close();

        for (int n : ene) {
            for (string t : tipo) {
                for (string d : dominio) {
                    for (string m : muestra) {
                        // se leen las matrices
                        string base = to_string(n) + "_" + t + "_" + d + "_" + m;
                        vector<vector<int>> mA = leer_matriz_txt("data/matrix_input/" + base + "_1.txt", n);
                        vector<vector<int>> mB = leer_matriz_txt("data/matrix_input/" + base + "_2.txt", n);
                        cout << "Procesando: " << n << " " << t << " " << d << " " << m << endl;

                        IterationMetrics metrics = run_iteration_isolated(
                            algo, n, t, d, m, mA, mB
                        );

                        cout << "Size: " << n
                             << " Time: " << fixed << setprecision(3) << metrics.time_ms << " ms"
                             << " RSS: " << metrics.rss_before_kb << " -> " << metrics.rss_after_kb << " kB"
                             << " DeltaRSS: " << metrics.delta_rss_kb << " kB"
                             << " Peak(iter): " << metrics.peak_kb << " kB"
                             << " DeltaPeakDesdeInicioIter: " << metrics.delta_peak_from_start_kb << " kB\n";
                        outfile.open("data/measurements/" + algo + ".csv", ios::app);
                        outfile << n << "," << t << "," << d << "," << m << ","
                            << fixed << setprecision(3) << metrics.time_ms << ","
                            << metrics.rss_before_kb << "," << metrics.rss_after_kb << "," << metrics.delta_rss_kb << ","
                            << metrics.peak_kb << "," << metrics.delta_peak_from_start_kb << "\n";
                        outfile.close();
                    }
                }
            }
        }
        add_completed(algo);
    }
    return 0;
}