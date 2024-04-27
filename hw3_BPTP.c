/*
    Evan Teschko
    916055702 
    Assignment 3
    To compile: mpicc -o hw3_BPTP hw3_BPTP.c
    To run: ./hw3_BPTP <size> <generations> 
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

double gettime(void) {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    return (double)tval.tv_sec + (double)tval.tv_usec / 1000000.0;
}

void freeBoard(int **board, int rows) {
    if (board != NULL) {
        if (rows > 0) {
            free(board[0]);
        }
        free(board);
    }
}

int **allocarray(int P, int Q) {
    int i;
    int *p, **a;
    p = (int *)malloc(P * Q * sizeof(int));
    a = (int **)malloc(P * sizeof(int *));
    if (p == NULL || a == NULL) {
        fprintf(stderr, "Error allocating memory\n");
        exit(-1);
    }
    for (i = 0; i < P; i++) {
        a[i] = &p[i * Q];
    }
    return a;
}

int countLiveNeighbourCell(int **a, int row, int col, int r, int c) {
    int i, j, count = 0;
    for (i = r-1; i <= r+1; i++) {
        for (j = c-1; j <= c+1; j++) {
            if (!(i == r && j == c) && i >= 0 && i < row && j >= 0 && j < col) {
                count += a[i][j];
            }
        }
    }
    return count;
}

int **giveBoardLife(int **original, int mrows, int ncols) {
    int i, j;
    for (i = 0; i < mrows; i++)
        for (j = 0; j < ncols; j++)
            original[i][j] = (drand48() > 0.5) ? 1 : 0;
    return original;
}

void printBoard(int **board, int rows, int cols, int rank, int size) {
    int *nums = NULL;
    if (rank == 0) {
        nums = malloc(rows * cols * sizeof(int));
    }
    MPI_Gather(board[0], rows * cols, MPI_INT, nums, rows * cols, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Board at generation:\n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%d ", nums[i * cols + j]);
            }
            printf("\n");
        }
        printf("\n");
        free(nums);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) fprintf(stderr, "Usage: %s <gridSize> <maxGen>\n", argv[0]);
        MPI_Finalize();
        exit(-1);
    }

    int gridSize = atoi(argv[1]);
    int maxGen = atoi(argv[2]);
    int ghostSize = gridSize + 2;
    int **b0 = allocarray(gridSize, gridSize);
    int **b1 = allocarray(gridSize, gridSize);
    int **b2 = allocarray(ghostSize, ghostSize);
    int **b3 = allocarray(ghostSize, ghostSize);

    if (rank == 0) {
        giveBoardLife(b0, gridSize, gridSize);
    }

    //Broadcast the initial grid state from the root to all other processes
    //so they all start with the same grid configuration for the simulation.
    MPI_Bcast(&(b0[0][0]), gridSize * gridSize, MPI_INT, 0, MPI_COMM_WORLD);

    double starttime = gettime();
    int currGen = 1;
    int changeDetected = 1;

    while (currGen < maxGen && changeDetected == 1) {
        //backup of b0 to b1
        for (int i = 0; i < gridSize; i++) {
            for (int j = 0; j < gridSize; j++) {
                b1[i][j] = b0[i][j];
            }
        }

        //Fill b2 from b0
        for (int i = 0; i < gridSize; i++) {
            for (int j = 0; j < gridSize; j++) {
                b2[i+1][j+1] = b0[i][j];
            }
        }

        //Handling of ghost cells
        for (int i = 1; i <= gridSize; i++) {
            b2[i][0] = b2[i][gridSize];
            b2[i][ghostSize-1] = b2[i][1];
        }

        //Exchange boundary rows with adjacent processes to update ghost cells
        if (size > 1) {
            MPI_Sendrecv(&b2[1][0], ghostSize, MPI_INT, (rank - 1 + size) % size, 0,
                         &b2[ghostSize-1][0], ghostSize, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&b2[gridSize][0], ghostSize, MPI_INT, (rank + 1) % size, 1,
                         &b2[0][0], ghostSize, MPI_INT, (rank - 1 + size) % size, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        //Computes next gen
        for (int i = 1; i <= gridSize; i++) {
            for (int j = 1; j <= gridSize; j++) {
                int liveNeighbors = countLiveNeighbourCell(b2, ghostSize, ghostSize, i, j);
                b3[i][j] = (b2[i][j] == 1 && (liveNeighbors == 2 || liveNeighbors == 3)) || (b2[i][j] == 0 && liveNeighbors == 3);
            }
        }

        //Copy b3 back to b0
        for (int i = 0; i < gridSize; i++) {
            for (int j = 0; j < gridSize; j++) {
                b0[i][j] = b3[i+1][j+1];
            }
        }

        //Gather and distribute the updated grid states from all processes,
        // ensuring every process has the complete grid information for the next iteration.
        MPI_Allgather(MPI_IN_PLACE, gridSize*gridSize/size, MPI_INT, b0[0], gridSize*gridSize/size, MPI_INT, MPI_COMM_WORLD);

        #ifdef DEBUG_PRINT
        if (rank == 0) {
            printBoard(b0, gridSize, gridSize, rank, size);
        }
        #endif
        
        //Checks to see if the board changed after a generation.
        int localChangeDetected = 0;
        for (int i = 0; i < gridSize; i++) {
            for (int j = 0; j < gridSize; j++) {
                if (b0[i][j] != b1[i][j]) {
                    localChangeDetected = 1;
                    break;
                }
            }
            if (localChangeDetected) break;
        }

        //Perform a global logical OR reduction to determine if a process has detected a change,
        // allowing all processes to decide to continue.
        MPI_Allreduce(&localChangeDetected, &changeDetected, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        currGen++;
    }

    double endtime = gettime();

    if (rank == 0) {
        printf("Final generation = %d\n", currGen);
        printf("Total time taken = %lf seconds\n", endtime - starttime);
    }

    freeBoard(b0, gridSize);
    freeBoard(b1, gridSize);
    freeBoard(b2, ghostSize);
    freeBoard(b3, ghostSize);

    MPI_Finalize();
    return 0;
}
