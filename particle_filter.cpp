/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"
#define EPS 0.0001

using namespace std;
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;

    // Gaussian noise
    normal_distribution<double> N_gps_x(x, std[0]);
    normal_distribution<double> N_gps_y(y, std[1]);
    normal_distribution<double> N_gps_theta(theta, std[2]);

    // Initialize the particles vector
    Particle p_tmp;
    for (int i=0; i<num_particles; ++i) {
        p_tmp.id = i;
        p_tmp.theta = theta;
        p_tmp.weight = 1.0;

        // add noise
        p_tmp.x = N_gps_x(gen);
        p_tmp.y = N_gps_y(gen);
        p_tmp.theta = N_gps_theta(gen);

        // normalize the angle to [-pi,pi)
        p_tmp.theta = fmod(p_tmp.theta + M_PI, (2.0*M_PI)) - M_PI;

        particles.push_back(p_tmp);
        weights.push_back(1.0);
    }

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    // Gaussian noise
    normal_distribution<double> N_pos_x(0, std_pos[0]);
    normal_distribution<double> N_pos_y(0, std_pos[1]);
    normal_distribution<double> N_pos_theta(0, std_pos[2]);

    if ((yaw_rate<EPS) && (yaw_rate>-EPS)){
        if (yaw_rate<0.0){
            yaw_rate=-EPS;
        }
        else{
            yaw_rate=EPS;
        }
    }

    double v_div_yaw_rate = velocity/yaw_rate;
    double dt_mul_yaw_rate = yaw_rate*delta_t;
    for (int i=0; i<num_particles; ++i){
        // calculate new position
        particles[i].x += v_div_yaw_rate * (sin (particles[i].theta + dt_mul_yaw_rate) - sin(particles[i].theta));
        particles[i].y += v_div_yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta+dt_mul_yaw_rate));
        particles[i].theta += dt_mul_yaw_rate;

        // add noise
        particles[i].x += N_pos_x(gen);
        particles[i].y += N_pos_y(gen);
        particles[i].theta += N_pos_theta(gen);

        // normalize the angle to [-pi,pi)
        particles[i].theta = fmod(particles[i].theta + M_PI, (2.0*M_PI)) - M_PI;
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	double min_dist, min_dist_test, dx, dy;
	int arg_min_dist;
	for (unsigned int i=0; i<observations.size(); ++i){
        arg_min_dist=predicted[0].id;
        dx = observations[i].x - predicted[0].x;
        dy = observations[i].y - predicted[0].y;
        min_dist = dx*dx + dy*dy;
        for (unsigned int j=1; j<predicted.size(); ++j){
            dx = observations[i].x - predicted[j].x;
            dy = observations[i].y - predicted[j].y;
            min_dist_test = dx*dx + dy*dy;
            if (min_dist_test<min_dist){
                arg_min_dist=predicted[j].id;
                min_dist = min_dist_test;
            }
        }
        observations[i].id=arg_min_dist;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	double sensor_range_sqr = sensor_range*sensor_range;
	LandmarkObs landOb_tmp;
	for (int i=0; i<num_particles; ++i){

        vector<LandmarkObs> predicted;
        double p_x = particles[i].x;
        double p_y = particles[i].y;
        double p_cos_t = cos(particles[i].theta);
        double p_sin_t = sin(particles[i].theta);
        for (unsigned int j=0; j<map_landmarks.landmark_list.size(); ++j){
            double mapOb_x = map_landmarks.landmark_list[j].x_f;
            double mapOb_y = map_landmarks.landmark_list[j].y_f;
            double dx = (p_x-mapOb_x);
            double dy = (p_y-mapOb_y);
            if ((dx*dx + dy*dy)<=sensor_range_sqr){
                landOb_tmp.id = map_landmarks.landmark_list[j].id_i;
                landOb_tmp.x = mapOb_x;
                landOb_tmp.y = mapOb_y;
                predicted.push_back(landOb_tmp);
            }
        }

        vector<LandmarkObs> observations_on_map;
        for (unsigned int j=0; j<observations.size(); ++j){
            landOb_tmp.id = -1;
            landOb_tmp.x = p_x + observations[j].x*p_cos_t - observations[j].y*p_sin_t;
            landOb_tmp.y = p_y + observations[j].x*p_sin_t + observations[j].y*p_cos_t;
            observations_on_map.push_back(landOb_tmp);
        }

        dataAssociation(predicted, observations_on_map);


        double log_weight = 0.0;
        double sigx2_inv = 0.5 / (std_landmark[0]*std_landmark[0]);
        double sigy2_inv = 0.5 / (std_landmark[1]*std_landmark[1]);
        for (unsigned int j=0; j<observations_on_map.size(); ++j){
            for (unsigned int k=0; k<predicted.size(); ++k){
                if (observations_on_map[j].id==predicted[k].id){
                    double dx = observations_on_map[j].x - predicted[k].x;
                    double dy = observations_on_map[j].y - predicted[k].y;
                    log_weight += sigx2_inv*dx*dx + sigy2_inv*dy*dy;
                    break;
                }
            }
        }
        weights[i] = exp(-log_weight);
        particles[i].weight = weights[i];
	}


}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> resamp_particles;
	double max_w = *max_element(weights.begin(), weights.end());

	uniform_int_distribution<int> Rnd_idx(0, num_particles-1);
	uniform_real_distribution<double> Rnd_w(0, 2.0*max_w);

	int idx = Rnd_idx(gen);
	double beta = 0.0;
	for (int i=0; i<num_particles; ++i){
        beta += Rnd_w(gen);
        while (weights[idx] < beta){
            beta -= weights[idx];
            idx = (idx+1) % num_particles;
        }
        resamp_particles.push_back(particles[idx]);
	}

    particles = resamp_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
