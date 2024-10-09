# include<stdlib.h>
# include<stdio.h>
# include<math.h>
# include<time.h>
# include<fcntl.h>
# include<unistd.h>
# include<stdint.h>
# include<omp.h>

# define S 100 //number of species evolving
# define T 2000000
    // storage limit: S*N = 10^8 would out put data of 1GB 
# define delay 0
# define N_sample 1000
# define N_matrix 10
# define LEN 200 // the length of ourput trajectory (only when write_traj_judge used)
# define NUM_BYTES sizeof(uint32_t)

// model parameters
double mu = 4; //average value of a_ij
double sig = 1; //variance of a_ij
double mygamma = 0; //covariance factor of a_ij and a_ji
double r; //= 2.5; // uniform growth rate of species.
// parameters to judge convergence
int N;
int interval = (1000 > (T/100)) ? 1000:(T/100); // time interval for convergence judgement
//double range = 0.00000000000001; // criteria for convergence
double range = 0.00000000000001;

int main(){
    // function claims
    void pairgenerate(double mean,double variance, double correlation, double* result1, double* result2); //function to generate correlated random number pair
    int matrix_generate(double a[S][S], double sig);
    int evolve(double** Nt, double a[S][S]);
    int converge(double** Nt);
    int single_time_check(double** Nt, int t, int back);
    void write_traj(double** Nt);
    int difference_judge(double** Nr, double** Ns, int pattern_r, int pattern_s);
    N = T;
    //printf("r=%F, mu=%F, sig=%F, T=%d, delay=%d, S=%d\n",r, mu, sig, T, delay, S);

    //allocate memory space for huge array Nt. Nt[i][t] is the abundance of species i at time t
    double a[S][S]; // array to store matrix A
    if (matrix_generate(a, sig)== -1){
        return 0;
    }

    char cname[50]; // spring for pattern file name
    sprintf(cname, "c_S%d_T%.0e_del%d_mu%.1e_sig%.1e.csv", S, (double)T, delay, mu, sig);
    FILE *cfile = fopen(cname, "w");
    // write in the time evolution data
    fprintf(cfile, "Numbe,Count,Pattern\n");

    char pname[50]; // spring for pattern traj file name
    sprintf(pname, "p_S%d_T%.0e_mu%.1e_sig%.1e.csv", S, (double)T, mu, sig);
    FILE *pfile = fopen(pname, "w");
    fprintf(pfile, "r,index,pattern,time,");
    for(int s = 1; s <100; s++){
        fprintf(pfile, "S%d,", s);
    }
    fprintf(pfile, "S100\n");

    for(r = 1; r < 2.3; r+=0.1){
    // for loop going over different values of r
        double ***Np; // large 3D array to record pattern trajectory Np[N_sample][Species][Time]
        Np = (double ***)malloc(N_sample * sizeof(double **));
        for(int sample = 0; sample < N_sample; sample++){
            Np[sample] = (double **)malloc(S * sizeof(double **));
        }
        int pattern[N_sample];// array to record the pattern of each sample
     
#pragma omp parallel for
        for(int sample = 0; sample < N_sample; sample++){
            double **Nt;// large 2D array for evolution
            Nt = (double **)malloc(S * sizeof(double *));
            for (int b = 0; b < S; b++) {
                Nt[b] = (double *)malloc(interval * sizeof(double));
            }

            if(evolve(Nt, a)){
                pattern[sample] = -2; //error code -2 for abnormal evolution
            }
            
            if(converge(Nt)){
                pattern[sample] = 0; // 0 for chaos, default value if no pattern found
                for (int o = 1; o < LEN; o++){
                    if(single_time_check(Nt, N-1, o)){
                        int back;
                        for (back = 1; back < LEN; back++){
                            if(single_time_check(Nt, N-1-back, o)){
                                continue;
                            }
                            break;
                        }
                        if(back == LEN){
                            pattern[sample] = o;
                            break;
                        }
                    }
                }
            }else{
                pattern[sample] = 1;// converged sample, pattern=1
            }

            if(pattern[sample]){
                for(int a = 0; a < S; a++){
                    Np[sample][a] = (double*)malloc(pattern[sample] * sizeof(double));
                    // allocate memory for non-chaotic samples to record their patterns
                }
                for(int s = 0; s < S; s++){
                    for(int a = 0; a < pattern[sample]; a++){
                        Np[sample][s][a] = Nt[s][(int)(N-1-pattern[sample]+a)%interval];
                    }
                }
            }else{// if no pattern, show a clip of LEN steps
                for(int a = 0; a < S; a++){
                    Np[sample][a] = (double*)malloc(LEN * sizeof(double));
                }
                for(int s = 0; s < S; s++){
                    for(int a = 0; a < LEN; a++){
                        Np[sample][s][a] = Nt[s][(int)(N-1-pattern[sample]+a)%interval];
                    }
                }
            }
            // release the memory reserved for array Nt
            
            for (int i = 0; i < S; i++) {
                free(Nt[i]);
            }
            free(Nt);
        }

        int count = 0; // variable recording total number of patterns found
        int patterns[N_sample]; // array to index different patterns for each sample, e.g. sample5 has the same pattern as sample1, then patterns[5] = 1, the index is the smallest sample that shows this patten
        patterns[0] = 0;

        for(int n = 1; n < N_sample; n++){
            int j = 0;
            for(; j < n; j++){
                if(difference_judge(Np[j], Np[n], pattern[j], pattern[n])){
                    continue;
                }
                patterns[n] = patterns[j];
                j--;
                break;
            }
            if(j == n){
                patterns[n] = ++count;
            }
        }

        int result[count + 1];// array to record how many samples are in each index
        int back_pointer[count + 1]; // array pointing to a sample of each pattern
        for(int i = 0; i < count + 1; i++){
            result[i] = 0;
        }
        for(int n = 0; n < N_sample; n++){
            result[patterns[n]]++;
            back_pointer[patterns[n]] = n;
        }
        /*
        printf("%d patterns found in total, with count as follows:\n", count + 1);
        for(int i = 0; i < count + 1; i++){
            printf("pattern %d: %d\n", i, result[i]);
        }
        */
        fprintf(cfile, "r=%.1e\n",r);
        for(int c = 0; c < count + 1; c++){
            fprintf(cfile, "%d,%d,%d\n", c+1, result[c], pattern[c]);
            if(c < 10){
                if(pattern[c]){
                    for(int t = 0; t < pattern[c]; t++){
                        fprintf(pfile, "%.1e,%d,%d,%d,", r, c + 1, pattern[c], t + 1);
                        for(int s = 0; s < S; s++){
                            fprintf(pfile, "%.15f,", Np[back_pointer[c]][s][t]);
                        }
                        fprintf(pfile, "\n");
                    }
                }else{
                    for(int t = 0; t < LEN; t++){
                        fprintf(pfile, "%.1e,%d,%d,%d,", r, c + 1, pattern[c], t + 1);
                        for(int s = 0; s < S; s++){
                            fprintf(pfile, "%.15f,", Np[back_pointer[c]][s][t]);
                        }
                    fprintf(pfile, "\n");
                    }
                }
            }
        }

        for(int sample = 0; sample < N_sample; sample++){
            for(int s = 0; s < S; s++){
                free(Np[sample][s]);
            }
            free(Np[sample]);
        }
        free(Np);
    }
    fclose(cfile);
    fclose(pfile);
    return 0;
}

// Function to generate related Gaussian random numbers (Acknowledgement: this function is originally generated by ChatGPT3.5 and further modefied by the author)
void pairgenerate(double mean, double variance, double correlation, double* result1, double* result2) {
    // Generate independent standard normal random numbers using Box-Muller transform
    double u1 = ((double)((long long int)rand()+1) / ((long long int)RAND_MAX+1));
    double u2 = ((double)((long long int)rand()+1) / ((long long int)RAND_MAX+1));
    double z1 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    double z2 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
    // Linear transformation to obtain correlated Gaussian random numbers
    *result1 = mean + sqrt(variance) * z1;
    *result2 = mean + (correlation  * (z1)) + sqrt(variance - correlation * correlation) * z2;
}

int matrix_generate(double a[S][S], double sig){
    void pairgenerate(double mean,double variance, double correlation, double* result1, double* result2); //function to generate correlated random number pair
    char aname[50]; // string as file name to store matrix A
    sprintf(aname, "a_S%d_mu%.0f_sig%.0f_gamma%.0f.csv", S, mu, sig, mygamma);
    //printf("Do you want to load an existing matrix?(1 for yes, 0 for no)\n");// to ask if the program should generate a nex random matrix or loading an existing one from a file.
    int choice; // parameter to show the choice
    //scanf("%d", &choice);
    choice = 0;
    if (choice == 0){ // to generate a new random matrix
        double rd1, rd2;// variable pair to store random number pair generated
        // go over the matrix and set values for a_ij
        srand((unsigned int)time(NULL));
        for(int i=0; i<S; i++){
            for (int j=0; j<i; j++){
                pairgenerate(0, (double)1/S, mygamma/S, &rd1, &rd2);
                a[i][j] = mu/S + sig*rd1;
                a[j][i] = mu/S + sig*rd2;
            }
            a[i][i] = 1;
        }
        // store the matrix generated in a CSV file for future use
        
        FILE *afile = fopen(aname, "w");
        for(int i=0; i<S; i++){
            for(int j=0; j<S; j++){
                fprintf(afile, "%.15f", a[i][j]);
                if (j < S-1){
                    fprintf(afile, ",");
                }
            }
            fprintf(afile, "\n");
        }
        fclose(afile);
    } else if (choice == 1){ // to load in an existing matrix
        FILE *efile = fopen(aname, "r");
        if(efile == NULL){ // error report for failing to open/find the file
            printf("Unable to load file %s\n", aname);
            return -1;
        }
        // load in the values of a_ij
        int i = 0, j = 0;
        char comma;
        char ret = ',';
        while (ret != EOF && i<S){
            for(j=0; j<S; j++){
                fscanf(efile, "%lf", &a[i][j]);
                ret = fscanf(efile, "%c", &comma);
            }
            i++;
        }
        fclose(efile);
    } else { // to warn the user that there is an invalid input
        printf("Invalid input, program ends\n");
        return -1;
    }
    return 0;
}

int evolve( double** Nt, double a[S][S]){
    double convertBytesToDouble(const unsigned char *bytes);
    for(int s = 0; s<S; s++){
        for(int t =0; t<interval; t++){
            Nt[s][t] = 0;
        }
    }
   // generate the initial values of each species

    srand((unsigned int)time(NULL)); // set the random seed again
    for(int s=0; s<S; s++){
        Nt[s][0] = 0.1 * (double)rand() / RAND_MAX; // random initial value between 0 and 1
        //printf("N%d=%Lf\n",s, Nt[s][0]);
    }
    
    /*
    int fd;
    unsigned char random_bytes[NUM_BYTES];
    fd = open("/dev/random", O_RDONLY);
    for (int s = 0; s < S; s++){
        read(fd, random_bytes, NUM_BYTES);
        Nt[s][0] = convertBytesToDouble(random_bytes) * 0.1;
    }
    close(fd);
    */
    // Evolution according to LV model
    int t = 1;
    int Ext[S]; // array to record if the species have been extinct
    int toExt[S];
    int change = 0;
    for(int i=0; i<S; i++){
        Ext[i] = 1;
        toExt[i] = 1;
    }
    // In the case of time delay, we ignore the competition effect before the delay data exist
    for(;t < delay + 1; t++){
        for (int i=0; i<S; i++){
            Nt[i][t] = (1+r)*Nt[i][t-1];
        }
    }
    register double sum; // put sum into register to speed up
    for(; t<N; t++){
    for(int i=0; i<S; i++){
        if(Ext[i]){
        // sum over the competition effects
        sum = 0.0;
        for(int j=0; j<S; j++){
            if(Ext[j]){
                sum += a[i][j]*Nt[j][(int) (t-1-(int) (delay))%interval]*Nt[i][(int) (t-1)%interval];    
        }}
        // evolution
        Nt[i][(int) t%interval] = Nt[i][(int) (t-1)%interval]+ r*(-sum + Nt[i][(int) (t-1)%interval]);
        // Kill the program when abnormal behaviour of N_i is found and report the situation
        if (Nt[i][(int) t%interval]>100000){ 
            //printf("blow up \n");
            printf("i=%d, N=%f", i, Nt[i][(int) t%interval]);
            return -1;
        }
        if (Nt[i][(int) t%interval]<range){
            //printf("error: negative abundance \n");
            //return -2;
            Nt[i][t%interval] = 0;
            toExt[i] = 0;
            change = 1;
        }
    }
        }
        if(change){
            for(int i=0; i<S; i++){
                Ext[i] = toExt[i];
            }
        }
        change = 0;
    }
    for(int s=0; s<S; s++){ // clean the traj of extinct species, whose array are stale as their evolution are stopped
        if(Ext[s]==0){
            for(int m = 0; m<interval; m++){
                Nt[s][m] = 0;
            }
        }
    }
    return 0;
}

int converge(double** Nt){
    // convergence judgement
    int status = 0; // variable to store the number of species that have not converged
    for (int s = 0; s<S; s++){
        for (int td = 0; td<interval; td++ ){
        if(Nt[s][(int) (N-1)%interval] - Nt[s][(int) (N-2 - td)%interval]>=range || Nt[s][(int)(N-2 - td)%interval] - Nt[s][(int)(N-1)%interval] >= range){// to see if the change on abundance of species i over the interval is within the range set before
            status++;
        }  
    }}
    return status;
}

void write_traj(double** Nt){
    char dname[50]; // spring for data file name
    sprintf(dname, "d_S%d_T%.0e_del%d_r%.1e.csv", S, (double)T, delay, r);
    FILE *dfile = fopen(dname, "w");
    // write in the time evolution data
    for (int k = 0; k<interval; k++){
        for (int l = 0; l<S; l++){
            fprintf(dfile, "%.15f", Nt[l][(N-interval+k)%interval]);
            if (l < S-1){
                fprintf(dfile, ",");
            }
        }
        fprintf(dfile, "\n");
    }
    fclose(dfile);
}

double convertBytesToDouble(const unsigned char *bytes){
    uint32_t result = 0;
    for (size_t i = 0; i < NUM_BYTES; i++){
        result |= bytes[i] << (8*i);
    }
    return (double)result/ ((double)UINT32_MAX + 1);
}

int single_time_check(double** Nt, int t, int back){
    int match = 0;
    for (int s = 0; s < S; s++){
        if(Nt[s][(int)t%interval] - Nt[s][(int)(t - back)%interval] < range && Nt[s][(int)(t - back)%interval] - Nt[s][(int)t%interval] < range){
            match++;
        }
    }
    if(match == S){
        return 1;
    }
    return 0;
}

int difference_judge(double** Nr, double** Ns, int pattern_r, int pattern_s){
    if(pattern_r != pattern_s || pattern_r == 0 || pattern_s == 0){
        return 1;
    }
    for(int offset = 0; offset < pattern_r; offset++){
        int t = 0;
        for(; t < pattern_r; t++){
            int s = 0;
            for(; s < S; s++){
                if(Nr[s][(t + offset)%pattern_r] - Ns[s][t] > range || Ns[s][t] - Nr[s][(t + offset)%pattern_r] > range){
                    s--;
                    break;
                }
            }
            if(s < S){
                t--;
                break;
            }
        }
        if(t == pattern_r){
            return 0;
        }
    }
    return 1;
}
