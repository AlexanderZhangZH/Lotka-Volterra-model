// This version is designed to be run on BRC, changing '|' to '_' in filename, resetting the random generation
# include<stdlib.h>
# include<stdio.h>
# include<math.h>
# include<time.h>
# include<fcntl.h>
# include<unistd.h>
# include<stdint.h>
# include<omp.h>

# define S 200 //number of species evolving
# define T 1000000
    // storage limit: S*N = 10^8 would out put data of 1GB 
# define delay 0
# define N_sample 8
# define N_matrix 8
# define LEN 200 // the length of ourput trajectory (only when write_traj_judge used)
# define NUM_BYTES sizeof(uint32_t)

// model parameters
long double mu = 4; //average value of a_ij
//long double sig = 0.1; //variance of a_ij
long double mygamma = 0; //covariance factor of a_ij and a_ji
long double r = 0.01;//= 2.5; // uniform growth rate of species.
// parameters to judge convergence
int N;
int interval = (1000 > (T/100)) ? 1000:(T/100); // time interval for convergence judgement
long double range = 0.00000000000001; // criteria for convergence

int main(){
    // function claims
    void pairgenerate(double mean, double variance, double correlation, double* result1, double* result2); //function to generate correlated random number pair
    int matrix_generate(long double** a, long double sig);
    int evolve(long double** Nt, long double** a);
    int converge(long double** Nt);
    void write_traj(long double** Nt);
    void write_samp(long double Ns[N_sample][S]);
    void write_traj_judge(long double** Nt, int sample);
    int sinlge_time_check(long double** Nt, int t, int back);
    void write_train(long double** Nt, FILE *tfile);

    // input model parameters
    //int T;
    //int delay;
    //printf("Input value of T\n");
    //scanf("%d", &T);
    //printf("Input value of delay\n");
    //scanf("%d", &delay);
    //delay = 0;
    //printf("Input valur of r\n");
    //scanf("%LF", &r);
    N = T;
    printf("r=%LF, mu=%LF, T=%d, delay=%d, S=%d, N_sample=%d\n",r, mu, T, delay, S, N_sample);

    // allocate memory space for huge array Nt. Nt[i][t] is the abundance of species i at time t
    /*
    long double **Nt;
    Nt = (long double **)malloc(S * sizeof(long double *));
    for (int b = 0; b < S; b++) {
        Nt[b] = (long double *)malloc(N * sizeof(long double));
    }
    */
    /*
    long double a[S][S]; // array to store matrix A
    if (matrix_generate(a)== -1){
        return 0;
    }
    */
    /*
    long double Ns[N_sample][S];
    int temp;
    int error = 0;
    for(int n = 0; n < N_sample; n++){
        temp = evolve( Nt, a);
        if ( temp !=0){
            error++;
            for(int s=0; s<S; s++){
                Ns[n][s] = temp;
            }
            continue;
        }
        long double sign = 1;
        if (converge(Nt) != 0){
            sign = -1;
        }
        for(int s=0; s<S; s++){
            Ns[n][s] = sign * Nt[s][N-1];
        }
    }  
    write_traj(Nt);
    */
    char tname[100];
    sprintf(tname, "t_S%d_T%0.e_r%.1Le_mu%.0Le.csv", S, (double)T, r, mu);
    FILE *tfile = fopen(tname, "w");
    long double sig = 1.5;
    for(; sig < 2.5; sig += 0.2){
        long long int SS = 0;
        long long int SS2 = 0;
        int error = 0;
#pragma omp parallel for
        for (int iter = 0; iter < N_matrix; iter++){
            srand((unsigned int)time(NULL) - iter*1000);
            /*
            long double a[S][S];
            matrix_generate(a, sig);
            */
            long double **a;
            a = (long double **)malloc(S* sizeof(long double*));
            for(int s = 0; s<S; s++){
                a[s] = (long double*)malloc(S* sizeof(long double));
            }
            matrix_generate(a, sig);

            long double **Nt;
            Nt = (long double **)malloc(S * sizeof(long double *));
            for (int b = 0; b < S; b++) {
                Nt[b] = (long double *)malloc((interval) * sizeof(long double));
            }

            int error_sub = 0;
            int temp;
            long long int SS_sub = 0;
            long long int SS2_sub = 0;

            for (int sample = 0; sample < N_sample/N_matrix; sample++){
                int SS_single = 0;
                temp = evolve(Nt, a);
                if(temp != 0){
                    error_sub++;
                    continue;
                }
                /*
                if(converge(Nt)){
                    error_sub++;
                    continue;
                }
                */
                //write_traj_judge(Nt, sample);
                //write_train(Nt, tfile);
                for (int s = 0; s < S; s++){
                    if(Nt[s][(N-1)%interval] > range){
                        SS_single++;
                    }
                }
                SS_sub += SS_single;
                SS2_sub += SS_single * SS_single;
            }
#pragma omp critical // Ensure only one thread accesses the file at a time
            {
                SS += SS_sub;
                SS2 += SS2_sub;
                error += error_sub;
            }
        for (int i = 0; i < S; i++) {
            free(Nt[i]);
            free(a[i]);
        }
        free(Nt);
        free(a);
        }
        fprintf(tfile, "%.1Le", sig);
        fprintf(tfile, ",");
        fprintf(tfile, "%d", error);
        fprintf(tfile, ",");
        fprintf(tfile, "%Le",(long double) SS/(N_sample - error));
        fprintf(tfile, ",");
        fprintf(tfile, "%Le",(long double) ((long double) SS2/(N_sample - error) - ((long double)SS/(N_sample - error))*((long double)SS/(N_sample - error))));
        fprintf(tfile, "\n");
        printf("%Lf \t %Le \t %Le \n", sig, (long double)SS/(N_sample - error), (long double) ((long double)SS2/(N_sample - error) - ((long double)SS/(N_sample - error))*((long double)SS/(N_sample - error))));
    }
    fclose(tfile);

    // release the memory reserved for array Nt
    /*
    for (int i = 0; i < S; i++) {
        free(Nt[i]);
    }
    free(Nt);
    */
    /*
    if (error == N_sample){
        printf("All samples wrong, sample 0 report: %Lf\n", Ns[0][0]);
        return 0;
    }
    */
    //write_samp(Ns);
    return 0;
}

// Function to generate related Gaussian random numbers (Acknowledgement: this function is originally generated by ChatGPT3.5 and further modefied by the author)
void pairgenerate(double mean, double variance, double correlation, double* result1, double* result2) {
    // Generate independent standard normal random numbers using Box-Muller transform
    double u1 = ((double)rand() / RAND_MAX);
    double u2 = ((double)rand() / RAND_MAX);
    double z1 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    double z2 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
    // Linear transformation to obtain correlated Gaussian random numbers
    *result1 = mean + sqrt(variance) * z1;
    *result2 = mean + (correlation  * (z1)) + sqrt(variance - correlation * correlation) * z2;
}

int matrix_generate(long double** a, long double sig){
    char aname[50]; // string as file name to store matrix A
    sprintf(aname, "a_S%d_mu%.0Lf_sig%.0Lf_gamma%.0Lf.csv", S, mu, sig, mygamma);
    //printf("Do you want to load an existing matrix?(1 for yes, 0 for no)\n");// to ask if the program should generate a nex random matrix or loading an existing one from a file.
    int choice; // parameter to show the choice
    //scanf("%d", &choice); 
    choice = 0;
    if (choice == 0){ // to generate a new random matrix
        double rd1, rd2;// variable pair to store random number pair generated
        // go over the matrix and set values for a_ij
        int i,j;
        //srand((unsigned int)time(NULL));
        for(i=0; i<S; i++){
            for (j=0; j<i; j++){
                pairgenerate(0, (long double)1/S, mygamma/S, &rd1, &rd2);
                a[i][j] = mu/S + sig*rd1;
                a[j][i] = mu/S + sig*rd2;
            }
            a[i][i] = 1;
        }
        // store the matrix generated in a CSV file for future use
        /*
        FILE *afile = fopen(aname, "w");
        for(i=0; i<S; i++){
            for(j=0; j<S; j++){
                fprintf(afile, "%Lf", a[i][j]);
                if (j < S-1){
                    fprintf(afile, ",");
                }
            }
            fprintf(afile, "\n");
        }
        fclose(afile);
        */
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
                fscanf(efile, "%Lf", &a[i][j]);
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

int evolve( long double** Nt, long double** a){
    double convertBytesToDouble(const unsigned char *bytes);
    for(int s = 0; s<S; s++){
        for(int t =0; t<interval; t++){
            Nt[s][t] = 0;
        }
    }
   // generate the initial values of each species
    
    int s = 0;
    srand((unsigned int)time(NULL)); // set the random seed again
    for(s=0; s<S; s++){
        Nt[s][0] = 0.1* ((double)rand() / RAND_MAX); // random initial value between 0 and 1
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
    int i = 0;
    int j = 0;
    int Ext[S]; // array to record if the species have been extinct
    int toExt[S];
    int change = 0;
    for(i=0; i<S; i++){
        Ext[i] = 1;
        toExt[S] = 1;
    }
    // In the case of time delay, we ignore the competition effect before the delay data exist
    for(;t < delay + 1; t++){
        for (i=0; i<S; i++){
            Nt[i][t] = (1+r)*Nt[i][t-1];
        }
    }
    register long double sum; // put sum into register to speed up
    for(; t<N; t++){
    for(i=0; i<S; i++){
        if(Ext[i]){
        // sum over the competition effects
        sum = 0.0;
        for(j=0; j<S; j++){
            if(Ext[j]){
                sum += a[i][j]*Nt[j][(int) (t-1-(int) (delay))%(interval)]*Nt[i][(int) (t-1)%(interval)];    
        }}
        // evolution
        Nt[i][(int) t%(interval)] = Nt[i][(int) (t-1)%(interval)] + r*(-sum + Nt[i][(int) (t-1)%(interval)]);
        // Kill the program when abnormal behaviour of N_i is found and report the situation
        if (Nt[i][(int) t%(interval)]>100000){ 
            //printf("blow up \n");
            return -1;
        }
        if (Nt[i][(int) t%(interval)]<0){
            //printf("error: negative abundance \n");
            //return -2;
            Nt[i][t%interval] = 0;
            toExt[i] = 0;
            change = 1;
        }
    }

        }
        if(change){
            for(i=0; i<S; i++){
                Ext[i] = toExt[i];
            }
        }
        change = 0;
        }
    return 0;
}

int converge(long double** Nt){
    // convergence judgement
    int status = 0; // variable to store the number of species that have not converged
    for (int s = 0; s<S; s++){
        for (int td = 0; td<interval; td++ ){
        if(Nt[s][(N-1)%(interval)] - Nt[s][td]>=range || Nt[s][td] - Nt[s][(N-1)%interval] >= range){// to see if the change on abundance of species i over the interval is within the range set before
            status++;
        }  
    }}
    return status;
}

void write_traj(long double** Nt){
    char dname[50]; // spring for data file name
    sprintf(dname, "d_S%d_T%.0e_del%d_r%.1Le.csv", S, (double)T, delay, r);
    FILE *dfile = fopen(dname, "w");
    // write in the time evolution data
    for (int k = 0; k<N; k++){
        for (int l = 0; l<S; l++){
            fprintf(dfile, "%Lf", Nt[l][k]);
            if (l < S-1){
                fprintf(dfile, ",");
            }
        }
        fprintf(dfile, "\n");
    }
    fclose(dfile);
}
/*
void write_samp(long double Ns[N_sample][S]){
    char sname[50]; // spring for data file name
    sprintf(sname, "s_S%d|T%.0e|del%d|r%.1Le|mu%.0Le|sig%.0Le.csv", S, (double)T, delay, r, mu, sig);
    FILE *sfile = fopen(sname, "w");
    // write in the time evolution data
    for (int k = 0; k<N_sample; k++){
        for (int l = 0; l<S; l++){
            fprintf(sfile, "%Lf", Ns[k][l]);
            if (l < S-1){
                fprintf(sfile, ",");
            }
        }
        fprintf(sfile, "\n");
    }
    fclose(sfile);
}

void write_traj_judge(long double** Nt, int sample){
    int single_time_check(long double** Nt, int t, int back);
    char jname[100];
    sprintf(jname, "s_S%d|T%0.e|r%.1Le|mu%.0Le|sig%.0Le_%d.csv", S, (double)T, r, mu, sig, sample);
    FILE *jfile = fopen(jname, "w");
     for (int l = 0; l<S; l++){
        fprintf(jfile, "%Lf", Nt[l][0]);
        if (l < S - 1){
            fprintf(jfile, ",");
        }
    }              
    fprintf(jfile, "\n");
    for (int k = N - LEN - 1; k<N; k++){
        for (int l = 0; l<S; l++){
            fprintf(jfile, "%Lf", Nt[l][k]);
            if (l < S - 1){
                fprintf(jfile, ",");
            }
        }              
        fprintf(jfile, "\n");
    }

    int pattern = 0;
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
                pattern = o;
                break;
            }
        }
    }
    int SS = 0;
    for (int s = 0; s < S; s++){
        if(Nt[s][N-1] != 0){
            SS++;
        }
    }
    
    //if(pattern != 0){
    //    fprintf(jfile, "%d", pattern);
    //    fprintf(jfile, "\n");
    //}
    
    fclose(jfile);
    char jname_new[100];
    sprintf(jname_new, "O%d_SS%d_S%d|T%0.e|r%.1Le|mu%.0Le|sig%.0Le_%d.csv", pattern, SS, S, (double)T, r, mu, sig, sample);
    rename(jname, jname_new);
}
*/
int single_time_check(long double** Nt, int t, int back){
    int match = 0;
    for (int s = 0; s < S; s++){
        if(Nt[s][t] - Nt[s][t - back] < range && Nt[s][t - back] - Nt[s][t] < range){
            match++;
        }
    }
    if(match == S){
        return 1;
    }
    return 0;
}
void write_train(long double** Nt, FILE *tfile){
    int single_time_check(long double** Nt, int t, int back);
    int pattern = 0;
    for (int l = 0; l<S; l++){
        fprintf(tfile, "%Lf", Nt[l][0]);
        fprintf(tfile, ",");
    }              
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
                pattern = o;
                break;
            }
        }
    }
    int SS = 0;
    for (int s = 0; s < S; s++){
        if(Nt[s][N-1] != 0){
            SS++;
        }
    }
    fprintf(tfile, "%d", pattern);
    fprintf(tfile, ",");
    fprintf(tfile, "%d", SS);
    fprintf(tfile, "\n");
}
double convertBytesToDouble(const unsigned char *bytes){
    uint32_t result = 0;
    for (size_t i = 0; i < NUM_BYTES; i++){
        result |= bytes[i] << (8*i);
    }
    return (double)result/ ((double)UINT32_MAX + 1);
}
