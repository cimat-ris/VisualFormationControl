#include <iostream>
#include "utils_vc.h"


using namespace std;

vc_state::vc_state() : X(0),Y(0),Z(0),Yaw(0),Pitch(0),Roll(0),
                      initialized(false),t(0),dt(0.025),Kv(1.0),Kw(1.0) {}

std::pair<Eigen::VectorXd,float> vc_state::update(const vc_control &command) {
  this->t+=this->dt;
  // Integrating
  this->X = this->X + this->Kv*command.Vx*this->dt;
  this->Y = this->Y + this->Kv*command.Vy*this->dt;
  this->Z = this->Z + this->Kv*command.Vz*this->dt;
  this->Yaw = this->Yaw + this->Kw*command.Vyaw*this->dt;
  cout << this->X << " " << this->Y << " " << this->Z << endl;

  Eigen::VectorXd position; position.resize(3);
  position(0) = this->X; position(1) = this->Y; position(2) = this->Z;
  return make_pair(position,this->Yaw);
}

void vc_state::initialize(const float &x,const float &y,const float &z,const float &yaw) {
  this->X = x;
  this->Y = y;
  this->Z = z;
  this->Yaw = yaw;
  initialized = true;
  cout << "Init pose" << endl << "X: " << this->X << endl << "Y: " << this->Y << endl << "Z: " << this->Z << endl;
  cout << "Roll: " << this->Roll << endl << "Pitch: " << this->Pitch << endl << "Yaw: " << this->Yaw << endl;
}