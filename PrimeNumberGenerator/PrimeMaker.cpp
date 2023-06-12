#include <fstream>  // Use a file
#include <iostream> // Other Output
#include <vector>   // List to store primes
#include <ciso646>  // To use the alternative operators
#include <string>   // For stoi
using namespace std;

string FILE_NAME = "Prime.txt";
int    NUM_PRIME = 10000;


bool isPrime(int p, vector<int> &primeList);

int main(int argc, char *argv[]) {
    // Uses default unless one is given
    int primeListSize = (argc > 1) ? stoi(argv[1]) : NUM_PRIME;
    string fileName   = (argc > 2) ? argv[2]       : FILE_NAME;

    ofstream file;
    file.open(fileName, ios::trunc);
    
    vector<int> primeList; 
    for (int i = 2; primeList.size() < primeListSize; i++) {
        if (not isPrime(i, primeList)) continue;
        
        primeList.push_back(i);
        file << i << endl;
    }

    file.close();
}

bool isPrime(int p, vector<int> &primeList) {
    for (int i : primeList)
        if (p % i == 0)
            return false;
    return true;
}