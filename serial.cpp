#include <iostream>
#include <fstream> 
#include <vector>
#include <math.h> 
#include <stdlib.h>
#include <algorithm>
#include <chrono> 

using namespace std;
using namespace std::chrono;

class Point {
// Attributes
private:
    int id_point, id_cluster;
    vector<double> values;
    int total_values;

public: // Constructor
    Point(int id_point, vector<double>& values) {
        this->id_point = id_point;
        total_values = values.size();

        for (int i = 0; i < total_values; i++)
            this->values.push_back(values[i]);

        id_cluster = -1;
    }
    // Getters&Setters
    int getID() { return id_point; }
    void setCluster(int id_cluster) { this->id_cluster = id_cluster; }
    int getCluster() { return id_cluster; }
    double getValue(int index) { return values[index]; }
    int getTotalValues() { return total_values; }
    void addValue(double value) { values.push_back(value); }
};

class Cluster {
private: // Attributes
    int id_cluster;
    vector<double> central_values;
    vector<Point> points;

public: 
    Cluster(int id_cluster, Point point) {  // Constructor
        this->id_cluster = id_cluster;

        int total_values = point.getTotalValues();

        for (int i = 0; i < total_values; i++)
            central_values.push_back(point.getValue(i));

        points.push_back(point);
    }

    void addPoint(Point point) { points.push_back(point); }

    bool removePoint(int id_point) {
        int total_points = points.size();

        for (int i = 0; i < total_points; i++) {
            if (points[i].getID() == id_point) {
                points.erase(points.begin() + i);
                return true;
            }
        }
        return false;
    }
    // Getters&Setters
    double getCentralValue(int index) { return central_values[index]; }
    void setCentralValue(int index, double value) { central_values[index] = value; }
    Point getPoint(int index) { return points[index]; } 
    int getTotalPoints() { return points.size(); }
    int getID() { return id_cluster; }
};

class KMeans {
private: // Attributes
    int K; // K is number of clusters
    int total_values, total_points, max_iterations;
    vector<Cluster> clusters;

    // return ID of nearest center (uses euclidean distance)
    int getIDNearestCenter(Point point) {
        double min_dist = INFINITY;
        int id_cluster_center = 0;

        for (int i = 0; i < K; i++) {
            double dist = 0.0;
            for (int j = 0; j < total_values; j++) {
                dist += pow(clusters[i].getCentralValue(j) - point.getValue(j), 2.0);
            }
            dist = sqrt(dist);

            if (dist < min_dist) {
                min_dist = dist;
                id_cluster_center = i;
            }
        }

        return id_cluster_center;
    }

    void assignPointsToClusters(vector<Point>& points) {
        for (int i = 0; i < total_points; i++) {
            int id_old_cluster = points[i].getCluster();
            int id_nearest_center = getIDNearestCenter(points[i]);

            if (id_old_cluster != id_nearest_center) {
                if (id_old_cluster != -1)
                    clusters[id_old_cluster].removePoint(points[i].getID());

                points[i].setCluster(id_nearest_center);
                clusters[id_nearest_center].addPoint(points[i]);
            }
        }
    }

public:
    KMeans(int K, int total_points, int total_values, int max_iterations) { // Constructor
        this->K = K;
        this->total_points = total_points;
        this->total_values = total_values;
        this->max_iterations = max_iterations;
    }

    void run(vector<Point>& points) { //  to execute the K-Means algorithm
        if (K > total_points)
            return;

        vector<int> base_indexes;

        // choose K distinct values for the centers of the clusters
        for (int i = 0; i < K; i++) {
            while (true) {
                int index_point = rand() % total_points;

                if (find(base_indexes.begin(), base_indexes.end(), index_point) == base_indexes.end()) {
                    base_indexes.push_back(index_point);
                    points[index_point].setCluster(i);
                    Cluster cluster(i, points[index_point]);
                    clusters.push_back(cluster);
                    break;
                }
            }
        }

        int iter = 1;

        while (true) {
            bool done = true;

            // Assign points to clusters
            assignPointsToClusters(points);

            // Recalculating the center of each cluster
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < total_values; j++) {
                    int total_points_cluster = clusters[i].getTotalPoints();
                    double sum = 0.0;

                    if (total_points_cluster > 0) {
                        for (int p = 0; p < total_points_cluster; p++) {
                            sum += clusters[i].getPoint(p).getValue(j);
                        }
                        clusters[i].setCentralValue(j, sum / total_points_cluster);
                    }
                }
            }

            // Check termination condition
            for (int i = 0; i < total_points; i++) {
                int id_old_cluster = points[i].getCluster();
                int id_nearest_center = getIDNearestCenter(points[i]);

                if (id_old_cluster != id_nearest_center) {
                    done = false;
                }
            }

            if (done || iter >= max_iterations) {
                cout << "Break in iteration " << iter << "\n\n";
                break;
            }

            iter++;
        }

        // Shows elements of clusters
        for (int i = 0; i < K; i++) {
            int total_points_cluster = clusters[i].getTotalPoints();

            cout << "Cluster " << clusters[i].getID() + 1 << endl;
            for (int j = 0; j < total_points_cluster; j++) {
                cout << "Point " << clusters[i].getPoint(j).getID() + 1 << ": ";
                for (int p = 0; p < total_values; p++)
                    cout << clusters[i].getPoint(j).getValue(p) << " ";

                cout << endl;
            }

            cout << "Cluster values: ";

            for (int j = 0; j < total_values; j++)
                cout << clusters[i].getCentralValue(j) << " ";

            cout << "\n\n";
        }
    }
};

int main() {
    srand(time(NULL));

    int total_points, total_values, K, max_iterations;

    // Read configuration from file
    ifstream config_file("config.txt");
    if (!config_file) {
        cerr << "Error: Could not open config file." << endl;
        return 1;
    }
    config_file >> total_points >> total_values >> K >> max_iterations;
    config_file.close();

    // Read points from file
    vector<Point> points;
    ifstream points_file("points.txt");
    if (!points_file) {
        cerr << "Error: Could not open points file." << endl;
        return 1;
    }

    for (int i = 0; i < total_points; i++) {
        vector<double> values;
        for (int j = 0; j < total_values; j++) {
            double value;
            points_file >> value;
            values.push_back(value);
        }
        Point p(i, values);
        points.push_back(p);
    }
    points_file.close();

    auto start = high_resolution_clock::now(); // Start timing

    KMeans kmeans(K, total_points, total_values, max_iterations);
    kmeans.run(points);

    auto end = high_resolution_clock::now(); // End timing
    auto duration = duration_cast<milliseconds>(end - start); // Calculate duration

    cout << "Execution time: " << duration.count() << " milliseconds" << endl;

    return 0;
}
