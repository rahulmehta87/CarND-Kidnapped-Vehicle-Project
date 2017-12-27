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
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    num_particles = 50;

    double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

    std_x = std[0];
    std_y = std[1];
    std_theta = std[2];

    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    default_random_engine gen;

    for (int i = 0; i < num_particles; i++) {
        double sample_x = dist_x(gen);
        double sample_y = dist_y(gen);
        double sample_theta = dist_theta(gen);
        std::vector<int> associations;
        std::vector<double> sense_x;
        std::vector<double> sense_y;

        Particle particle{i, sample_x, sample_y, sample_theta, 1, associations, sense_x, sense_y};

        particles.push_back(particle);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    default_random_engine gen;

    for (int i = 0; i < num_particles; i++) {
        double x_0 = particles[i].x;
        double y_0 = particles[i].y;
        double theta_0 = particles[i].theta;

        particles[i].theta = theta_0 + yaw_rate * delta_t;

        if (yaw_rate > 0.001) {
            double scaling_factor = velocity / yaw_rate;
            particles[i].x = x_0 + scaling_factor * (sin(particles[i].theta) - sin(theta_0));
            particles[i].y = y_0 + scaling_factor * (cos(theta_0) - cos(particles[i].theta));
        } else {
            particles[i].x = x_0 + velocity * delta_t * cos(theta_0);
            particles[i].y = y_0 + velocity * delta_t * sin(theta_0);
        }

        double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

        std_x = std_pos[0];
        std_y = std_pos[1];
        std_theta = std_pos[2];

        normal_distribution<double> dist_x(particles[i].x, std_x);
        normal_distribution<double> dist_y(particles[i].y, std_y);
        normal_distribution<double> dist_theta(particles[i].theta, std_theta);

        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    for (int i = 0; i < observations.size(); i++) {
        double old_distance_sq = -1;
        for (int j = 0; j < predicted.size(); j++) {
            LandmarkObs landmark = predicted[j];

            if (observations[i].id == 0) {
                observations[i].id = landmark.id;
                old_distance_sq = (landmark.x - observations[i].x) * (landmark.x - observations[i].x)
                                       + (landmark.y - observations[i].y) * (landmark.y - observations[i].y);
            } else {
                double new_distance_sq = (landmark.x - observations[i].x) * (landmark.x - observations[i].x)
                                       + (landmark.y - observations[i].y) * (landmark.y - observations[i].y);

                if (new_distance_sq < old_distance_sq) {
                    observations[i].id = landmark.id;
                    old_distance_sq = new_distance_sq;
                }
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    double sigma_x = std_landmark[0];
    double sigma_y = std_landmark[1];

    for (int i = 0; i < num_particles; i++) {
        std::vector<LandmarkObs> predicted;
        std::vector<LandmarkObs> observations_m;

        for (int j = 0; j < observations.size(); j++) {
            double x_c = observations[j].x;
            double y_c = observations[j].y;

            double x_m = particles[i].x + x_c * cos(particles[i].theta) - y_c * sin(particles[i].theta);
            double y_m = particles[i].y + x_c * sin(particles[i].theta) + y_c * cos(particles[i].theta);

            LandmarkObs observation_m = LandmarkObs{0, x_m, y_m};
            observations_m.push_back(observation_m);
        }

        for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            Map::single_landmark_s landmark = map_landmarks.landmark_list[j];

            double distance_sqr = (particles[i].x - landmark.x_f) * (particles[i].x - landmark.x_f)
                                + (particles[i].y - landmark.y_f) * (particles[i].y - landmark.y_f);

            if (distance_sqr < (sensor_range * sensor_range)) {
                LandmarkObs landmark_obs = LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f};
                predicted.push_back(landmark_obs);
            }
        }

        dataAssociation(predicted, observations_m);

        double weight = 1.0;
        std::vector<int> associations;
        std::vector<double> sense_x;
        std::vector<double> sense_y;

        for (int j = 0; j < observations_m.size(); j++) {
            double delta_x = observations_m[j].x - map_landmarks.landmark_list[observations_m[j].id - 1].x_f;
            double delta_y = observations_m[j].y - map_landmarks.landmark_list[observations_m[j].id - 1].y_f;
            double x_term = (delta_x) * (delta_x) / (2 * sigma_x * sigma_x);
            double y_term = (delta_y) * (delta_y) / (2 * sigma_y * sigma_y);
            double probability = exp(-(x_term + y_term)) / (2 * M_PI * sigma_x * sigma_y);

            weight = weight * probability;
            associations.push_back(observations_m[j].id);
            sense_x.push_back(observations_m[j].x);
            sense_y.push_back(observations_m[j].y);
        }

        particles[i].weight = weight;
        particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);
    }
}

void ParticleFilter::resample() {
    std::vector<double> weights;

    for (int i = 0; i < num_particles; i++) {
        weights.push_back(particles[i].weight);
    }

    std::discrete_distribution<> d(weights.begin(), weights.end());

    std::vector<Particle> new_particles;

    default_random_engine gen;

    for (int i = 0; i < num_particles; i++) {
        new_particles.push_back(particles[d(gen)]);
    }

    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
