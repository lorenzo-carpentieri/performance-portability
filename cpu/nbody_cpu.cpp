#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <chrono>

#define N 30208
#define DELTA_TIME 0.2f
#define SOFTENING 1e-9f

struct float4{            
    float x;
    float y;
    float z;
    float w;
};  

struct float3{            
    float x;
    float y;
    float z;
};  

 // operator overloading for floatX
    float4 operator*(float4 a, float4 b){
                float4 result;
                result.x = a.x * b.x;
                result.y = a.y * b.y;
                result.z = a.z * b.z;
                result.w = a.w * b.w;
                return result;
            }

    float4 operator+(float4 a, float4 b){
                float4 result;
                result.x = a.x + b.x;
                result.y = a.y + b.y;
                result.z = a.z + b.z;
                result.w = a.w + b.w;
                return result;
            }
    float3 operator+(float3 a, float4 b){
        float3 result;
        result.x = a.x + b.x;
        result.y = a.y + b.y;
        result.z = a.z + b.z;
        return result;
    }


    float4 operator-(float4 a, float4 b){
                float4 result;
                result.x = a.x - b.x;
                result.y = a.y - b.y;
                result.z = a.z - b.z;
                result.w = a.w - b.w;
                return result;
            }

    float4 operator*(float4 a, float b){
                float4 result;
                result.x = a.x * b;
                result.y = a.y * b;
                result.z = a.z * b;
                result.w = a.w * b;
                return result;
            }


void bodyForce(float4* pos, float4* vel, float4* new_pos, float4* new_vel) {
 
  for (int i = 0; i < N; i++) { 
    float4 acc4={0,0,0,0};

    for (int j = 0; j < N; j++) {
        float4 d = pos[j] - pos[i];
        float distSqr = d.x*d.x + d.y*d.y + d.z*d.z + SOFTENING;
        float invDist = 1.0f / sqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;
        float s = pos[j].w * invDist3;
        acc4 = acc4 + d * s;
    }

    float4 oldVel = vel[i];
    // updated position and velocity
    float4 newPos = pos[i] + oldVel * DELTA_TIME + acc4 * 0.5f * DELTA_TIME * DELTA_TIME;
    newPos.w = pos[i].w;
    // float4 newVel = oldVel + (acc4 * DELTA_TIME);

    new_pos[i] = newPos;
  
    new_vel[i] = vel[i]+(acc4 * DELTA_TIME);
    
  }
}

int main(const int argc, const char** argv) {
  
    
    const int nIters = 2;  // simulation iterations

    int bytes = N * sizeof(float4);

    float4 pos[N];
    float4 vel[N];

    float4 new_pos[N];
    float4 new_vel[N];

    // Initialization bodies pos and vel
    srand(10);
    for(int i = 0; i < N; i++){
        // pos
        pos[i].x= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        pos[i].y= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        pos[i].z= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        // mass
        pos[i].w= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        //vel
        vel[i].x= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        vel[i].y= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        vel[i].z= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        vel[i].w= 0;
       
    }
  float total_time = 0;
  for (int iter = 1; iter <= nIters; iter++) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    bodyForce(pos, vel, new_pos, new_vel); // compute interbody forces
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    for(int i = 0; i < N; i++){
        printf("Simulazione: %d, body_%d, new_pos_x: %.2f, new_pos_y: %.2f, new_pos_z: %.2f\n", iter, i, new_pos[i].x, new_pos[i].y, new_pos[i].z);
        printf("Simulazione: %d, body_%d, new_vel_x: %.2f, new_vel_y: %.2f, new_vel_z: %.2f\n", iter, i, new_vel[i].x, new_vel[i].y, new_vel[i].z);
        printf("\n");
    }
  }
  printf("Time average: %f", total_time/nIters);
    

  
}