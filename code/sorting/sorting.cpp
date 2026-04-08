#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include "algorithms/mergesort.cpp"
#include "algorithms/quicksort.cpp"
#include "algorithms/sort.cpp"
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

struct IterationMetrics {
    double time_ms;
    long rss_before_kb;
    long rss_after_kb;
    long delta_rss_kb;
    long peak_kb;
    long delta_peak_from_start_kb;
};

IterationMetrics invalid_metrics() {
    IterationMetrics m;
    m.time_ms = -1.0;
    m.rss_before_kb = -1;
    m.rss_after_kb = -1;
    m.delta_rss_kb = -1;
    m.peak_kb = -1;
    m.delta_peak_from_start_kb = -1;
    return m;
}

void guardar_arreglo(const vector<int>& arr, const string& outfilename) {
    ofstream outarr("data/array_output/" + outfilename);
    for (size_t i = 0; i < arr.size(); ++i) {
        outarr << arr[i];
        if (i + 1 < arr.size()) outarr << " ";
    }
    outarr.close();
}

bool check_completed(string algo) {
    ifstream infile("data/measurements/completion_list.txt");
    string linea;
    while (getline(infile, linea)) {
        if (linea == algo) return true;
    }
    return false;
}

void add_completed(string algo) {
    ofstream outfile("data/measurements/completion_list.txt", ios::app);
    outfile << algo << "\n";
    outfile.close();
}

IterationMetrics run_iteration_isolated(
    const string& algo,
    const vector<int>& input_arr,
    const string& outfilename
) {
    IterationMetrics metrics = invalid_metrics();
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

        // Intenta usar el maximo stack permitido para reducir fallos por recursividad.
        struct rlimit stack_limits {};
        if (getrlimit(RLIMIT_STACK, &stack_limits) == 0 &&
            stack_limits.rlim_cur < stack_limits.rlim_max) {
            stack_limits.rlim_cur = stack_limits.rlim_max;
            setrlimit(RLIMIT_STACK, &stack_limits);
        }

        vector<int> arr = input_arr;
        long rss_before = get_current_rss_kb();

        auto start = high_resolution_clock::now();
        if (algo == "mergesort") mergeSort(arr, 0, arr.size() - 1);
        else if (algo == "quicksort") quickSort(arr.data(), 0, arr.size() - 1);
        else if (algo == "sort") std::sort(arr.begin(), arr.end());
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

        guardar_arreglo(arr, outfilename);

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
        cerr << "Fallo al medir la iteracion en proceso aislado.";
        if (WIFSIGNALED(status)) {
            cerr << " Senal: " << WTERMSIG(status);
        } else if (WIFEXITED(status)) {
            cerr << " Exit code: " << WEXITSTATUS(status);
        }
        cerr << endl;
        metrics = invalid_metrics();
    }

    return metrics;
}

int main() {
    vector<int> ene = {10, 1000, 100000, 10000000};
    vector<string> tipo = {"ascendente", "descendente", "aleatorio"};
    vector<string> dominio = {"D1", "D7"};
    vector<string> muestra = {"a", "b", "c"};


    // todos los algoritmos
    vector<string> algoritmo = {"mergesort", "quicksort", "sort"};


    // creamos archivo .csv

    for (string algo : algoritmo) {

        if (check_completed(algo)) continue;

        ofstream outfile("data/measurements/" + algo + ".csv");
        outfile << "size,tipo,dominio,muestra,time_ms,rss_before_kb,rss_after_kb,delta_rss_kb,peak_kb,delta_peak_from_start_kb\n";

        cout << "----> " << algo << " <----" << endl;

        for (const int& n : ene) {
            for (const string& t : tipo) {
                for (const string& d : dominio) {
                    for (const string& m : muestra) {
                        
                        // todos lso archivos
                        string filename = "data/array_input/" + to_string(n) + "_" + t + "_" + d + "_" + m + ".txt";
                        
                        // Leer los datos del archivo
                        ifstream infile(filename);
                        vector<int> arr;
                        int num;
                        
                        // chequeo de los archivos
                        if (!infile) {
                            cerr << "No se pudo abrir: " << filename << endl;
                            continue;
                        }
                        while (infile >> num) {
                            arr.push_back(num);
                        }


                        infile.close();
                        cout << "Procesando: " << filename << " (" << arr.size() << " elementos)" << endl;


                        string outfilename = algo + " " + to_string(n) + "_" + t + "_" + d + "_" + m + "_out.txt";

                        IterationMetrics metrics = run_iteration_isolated(algo, arr, outfilename);

                        // lo guardamos
                        outfile << arr.size() << "," << t << "," << d << "," << m << ","
                            << fixed << setprecision(3) << metrics.time_ms << ","
                            << metrics.rss_before_kb << "," << metrics.rss_after_kb << ","
                            << metrics.delta_rss_kb << "," << metrics.peak_kb << ","
                            << metrics.delta_peak_from_start_kb << "\n";
                        outfile.flush();


                        // Mostrar el resultado
                        cout << "Size: " << arr.size()
                             << " Time: " << fixed << setprecision(3) << metrics.time_ms << " ms"
                             << " RSS: " << metrics.rss_before_kb << " -> " << metrics.rss_after_kb << " kB"
                             << " DeltaRSS: " << metrics.delta_rss_kb << " kB"
                             << " Peak(iter): " << metrics.peak_kb << " kB"
                             << " DeltaPeakDesdeInicioIter: " << metrics.delta_peak_from_start_kb << " kB\n";

                        
                    }
                }
            }
        }
        // se indica que se completo el procesado del algoritmo
        add_completed(algo);
    }

    

    return 0;
}