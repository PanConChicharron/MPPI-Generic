#include "tube_mppi_controller.cuh"
#include <mppi/core/mppi_common.cuh>

#define TUBE_MPPI_TEMPLATE template <class DYN_T, class COST_T, class FB_T, class SAMPLING_T, class PARAMS_T>

#define TubeMPPI TubeMPPIController<DYN_T, COST_T, FB_T, SAMPLING_T, PARAMS_T>

TUBE_MPPI_TEMPLATE
TubeMPPI::TubeMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller, SAMPLING_T* sampler, float dt,
                             int max_iter, float lambda, float alpha, int num_timesteps, int num_rollouts,
                             const Eigen::Ref<const control_trajectory>& init_control_traj, cudaStream_t stream)
  : PARENT_CLASS(model, cost, fb_controller, sampler, dt, max_iter, lambda, alpha, num_timesteps, num_rollouts,
                 init_control_traj, stream)
{
  // call rollout kernel with z = 2 since we have a nominal state
  this->params_.dynamics_rollout_dim_.z = max(2, this->params_.dynamics_rollout_dim_.z);
  this->params_.cost_rollout_dim_.z = max(2, this->params_.cost_rollout_dim_.z);
  this->sampler_->setNumDistributions(2);
  this->sample_multiplier_ = 2;

  // Properly size nominal variables
  setNumTimestepsHelper(this->getNumTimesteps(), false);
  setNumRolloutsHelper(this->getNumRollouts(), false);

  // Zero the nominal trajectories
  nominal_state_trajectory_.setZero();
  nominal_state_trajectory_.setZero();
  nominal_control_trajectory_ = this->params_.init_control_traj_;
  trajectory_costs_nominal_.setZero();

  // Allocate CUDA memory for the controller
  allocateCUDAMemory();

  // Initialize Feedback
  this->fb_controller_->initTrackingController();
  this->enable_feedback_ = true;
  chooseAppropriateKernel();
}

TUBE_MPPI_TEMPLATE
TubeMPPI::TubeMPPIController(DYN_T* model, COST_T* cost, FB_T* fb_controller, SAMPLING_T* sampler, PARAMS_T& params,
                             cudaStream_t stream)
  : PARENT_CLASS(model, cost, fb_controller, sampler, params, stream)
{
  // call rollout kernel with z = 2 since we have a nominal state
  this->params_.dynamics_rollout_dim_.z = max(2, this->params_.dynamics_rollout_dim_.z);
  this->params_.cost_rollout_dim_.z = max(2, this->params_.cost_rollout_dim_.z);
  this->sampler_->setNumDistributions(2);
  this->sample_multiplier_ = 2;

  // Properly size nominal variables
  setNumTimestepsHelper(this->getNumTimesteps(), false);
  setNumRolloutsHelper(this->getNumRollouts(), false);

  // Zero the nominal trajectories
  nominal_state_trajectory_.setZero();
  nominal_state_trajectory_.setZero();
  nominal_output_trajectory_.setZero();
  nominal_control_trajectory_ = this->params_.init_control_traj_;
  trajectory_costs_nominal_.setZero();

  // Allocate CUDA memory for the controller
  allocateCUDAMemory();

  // Initialize Feedback
  this->fb_controller_->initTrackingController();
  this->enable_feedback_ = true;
  chooseAppropriateKernel();
}

TUBE_MPPI_TEMPLATE
void TubeMPPI::chooseAppropriateKernel()
{
  cudaDeviceProp deviceProp;
  HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, 0));
  unsigned single_kernel_byte_size = mppi::kernels::calcRolloutCombinedKernelSharedMemSize(
      this->model_, this->cost_, this->sampler_, this->params_.dynamics_rollout_dim_);
  unsigned split_dyn_kernel_byte_size = mppi::kernels::calcRolloutDynamicsKernelSharedMemSize(
      this->model_, this->sampler_, this->params_.dynamics_rollout_dim_);
  unsigned split_cost_kernel_byte_size =
      mppi::kernels::calcRolloutCostKernelSharedMemSize(this->cost_, this->sampler_, this->params_.cost_rollout_dim_);
  unsigned vis_single_kernel_byte_size = mppi::kernels::calcVisualizeKernelSharedMemSize(
      this->model_, this->cost_, this->sampler_, this->getNumTimesteps(), this->params_.visualize_dim_);

  bool too_much_mem_single_kernel = single_kernel_byte_size > deviceProp.sharedMemPerBlock;
  bool too_much_mem_vis_kernel = vis_single_kernel_byte_size > deviceProp.sharedMemPerBlock;
  bool too_much_mem_split_kernel = split_dyn_kernel_byte_size > deviceProp.sharedMemPerBlock;
  too_much_mem_split_kernel = too_much_mem_split_kernel || split_cost_kernel_byte_size > deviceProp.sharedMemPerBlock;
  too_much_mem_single_kernel = too_much_mem_single_kernel || too_much_mem_vis_kernel;

  if (too_much_mem_split_kernel && too_much_mem_single_kernel)
  {
    std::string error_msg =
        "There is not enough shared memory on the GPU for either rollout kernel option. The combined rollout kernel "
        "takes " +
        std::to_string(single_kernel_byte_size) + " bytes, the cost rollout kernel takes " +
        std::to_string(split_cost_kernel_byte_size) + " bytes, the dynamics rollout kernel takes " +
        std::to_string(split_dyn_kernel_byte_size) + " bytes, the combined visualization kernel takes " +
        std::to_string(vis_single_kernel_byte_size) + " bytes, and the max is " +
        std::to_string(deviceProp.sharedMemPerBlock) +
        " bytes. Considering lowering the corresponding thread block sizes.";
    throw std::runtime_error(error_msg);
  }
  else if (too_much_mem_single_kernel)
  {
    this->setKernelChoice(kernelType::USE_SPLIT_KERNELS);
    return;
  }
  else if (too_much_mem_split_kernel)
  {
    this->setKernelChoice(kernelType::USE_SINGLE_KERNEL);
    return;
  }

  // Send the nominal control to the device
  this->copyNominalControlToDevice(false);
  state_array zero_state = this->model_->getZeroState();
  // Send zero state to the device
  HANDLE_ERROR(cudaMemcpyAsync(this->initial_state_d_, zero_state.data(), DYN_T::STATE_DIM * sizeof(float),
                               cudaMemcpyHostToDevice, this->stream_));
  HANDLE_ERROR(cudaMemcpyAsync(this->initial_state_d_ + DYN_T::STATE_DIM, zero_state.data(),
                               DYN_T::STATE_DIM * sizeof(float), cudaMemcpyHostToDevice, this->stream_));
  // Generate noise data
  this->sampler_->generateSamples(1, 0, this->gen_, true);

  float single_kernel_time_ms = std::numeric_limits<float>::infinity();
  float split_kernel_time_ms = std::numeric_limits<float>::infinity();

  // Evaluate each kernel that is applicable
  auto start_single_kernel_time = std::chrono::steady_clock::now();
  for (int i = 0; i < this->getNumKernelEvaluations() && !too_much_mem_single_kernel; i++)
  {
    mppi::kernels::launchRolloutKernel<DYN_T, COST_T, SAMPLING_T>(
        this->model_, this->cost_, this->sampler_, this->getDt(), this->getNumTimesteps(), this->getNumRollouts(),
        this->getLambda(), this->getAlpha(), this->initial_state_d_, this->trajectory_costs_d_,
        this->params_.dynamics_rollout_dim_, this->stream_, true);
  }
  auto end_single_kernel_time = std::chrono::steady_clock::now();
  auto start_split_kernel_time = std::chrono::steady_clock::now();
  for (int i = 0; i < this->getNumKernelEvaluations() && !too_much_mem_split_kernel; i++)
  {
    mppi::kernels::launchSplitRolloutKernel<DYN_T, COST_T, SAMPLING_T>(
        this->model_, this->cost_, this->sampler_, this->getDt(), this->getNumTimesteps(), this->getNumRollouts(),
        this->getLambda(), this->getAlpha(), this->initial_state_d_, this->output_d_, this->trajectory_costs_d_,
        this->params_.dynamics_rollout_dim_, this->params_.cost_rollout_dim_, this->stream_, true);
  }
  auto end_split_kernel_time = std::chrono::steady_clock::now();

  // calc times
  if (!too_much_mem_single_kernel)
  {
    single_kernel_time_ms = mppi::math::timeDiffms(end_single_kernel_time, start_single_kernel_time);
  }
  if (!too_much_mem_split_kernel)
  {
    split_kernel_time_ms = mppi::math::timeDiffms(end_split_kernel_time, start_split_kernel_time);
  }
  std::string kernel_choice = "";
  if (split_kernel_time_ms < single_kernel_time_ms)
  {
    this->setKernelChoice(kernelType::USE_SPLIT_KERNELS);
    kernel_choice = "split ";
  }
  else
  {
    this->setKernelChoice(kernelType::USE_SINGLE_KERNEL);
    kernel_choice = "single";
  }
  this->logger_->info("Choosing %s kernel based on split taking %f ms and single taking %f ms after %d iterations\n",
                      kernel_choice.c_str(), split_kernel_time_ms, single_kernel_time_ms,
                      this->getNumKernelEvaluations());
}

TUBE_MPPI_TEMPLATE
void TubeMPPI::setNumTimestepsHelper(const int num_timesteps, const bool update_gpu_mem)
{
  PARENT_CLASS::setNumTimestepsHelper(num_timesteps, update_gpu_mem);
  // Set up nominal trajectories
  PARENT_CLASS::resizeTimeTrajectory(nominal_control_trajectory_, num_timesteps);
  nominal_control_trajectory_ = this->params_.init_control_traj_;
  PARENT_CLASS::resizeTimeTrajectory(nominal_state_trajectory_, num_timesteps);
  PARENT_CLASS::resizeTimeTrajectory(nominal_output_trajectory_, num_timesteps);
}

TUBE_MPPI_TEMPLATE
void TubeMPPI::setNumRolloutsHelper(const int num_rollouts, const bool update_gpu_mem)
{
  PARENT_CLASS::setNumRolloutsHelper(num_rollouts, update_gpu_mem);
  // Set up nominal trajectories
  Eigen::NoChange_t same_col = Eigen::NoChange_t::NoChange;
  trajectory_costs_nominal_.conservativeResize(num_rollouts, same_col);
}

TUBE_MPPI_TEMPLATE
void TubeMPPI::computeControl(const Eigen::Ref<const state_array>& state, int optimization_stride)
{
  if (!nominalStateInit_)
  {
    // set the nominal state to the actual state
    nominal_state_trajectory_.col(0) = state;
    nominalStateInit_ = true;
  }

  this->free_energy_statistics_.real_sys.previousBaseline = this->getBaselineCost(0);
  this->free_energy_statistics_.nominal_sys.previousBaseline = this->getBaselineCost(1);

  //  std::cout << "Post disturbance Actual State: "; this->model_->printState(state.data());
  //  std::cout << "                Nominal State: "; this->model_->printState(nominal_state_trajectory_.col(0).data());

  // Handy reference pointers to the nominal state
  float* trajectory_costs_nominal_d = this->trajectory_costs_d_ + this->getNumRollouts();
  float* initial_state_nominal_d = this->initial_state_d_ + DYN_T::STATE_DIM;

  for (int opt_iter = 0; opt_iter < this->getNumIters(); opt_iter++)
  {
    // Send the initial condition to the device
    HANDLE_ERROR(cudaMemcpyAsync(this->initial_state_d_, state.data(), DYN_T::STATE_DIM * sizeof(float),
                                 cudaMemcpyHostToDevice, this->stream_));
    HANDLE_ERROR(cudaMemcpyAsync(initial_state_nominal_d, nominal_state_trajectory_.data(),
                                 DYN_T::STATE_DIM * sizeof(float), cudaMemcpyHostToDevice, this->stream_));

    // Send the nominal control to the device
    copyControlToDevice(false);

    // Generate noise data
    this->sampler_->generateSamples(optimization_stride, opt_iter, this->gen_, false);

    // call rollout kernel with z = 2 since we have a nominal state
    this->params_.dynamics_rollout_dim_.z = max(2, this->params_.dynamics_rollout_dim_.z);
    this->params_.cost_rollout_dim_.z = max(2, this->params_.cost_rollout_dim_.z);

    // Launch the rollout kernel
    if (this->getKernelChoiceAsEnum() == kernelType::USE_SPLIT_KERNELS)
    {
      mppi::kernels::launchSplitRolloutKernel<DYN_T, COST_T, SAMPLING_T>(
          this->model_, this->cost_, this->sampler_, this->getDt(), this->getNumTimesteps(), this->getNumRollouts(),
          this->getLambda(), this->getAlpha(), this->initial_state_d_, this->output_d_, this->trajectory_costs_d_,
          this->params_.dynamics_rollout_dim_, this->params_.cost_rollout_dim_, this->stream_, false);
    }
    else
    {
      mppi::kernels::launchRolloutKernel<DYN_T, COST_T, SAMPLING_T>(
          this->model_, this->cost_, this->sampler_, this->getDt(), this->getNumTimesteps(), this->getNumRollouts(),
          this->getLambda(), this->getAlpha(), this->initial_state_d_, this->trajectory_costs_d_,
          this->params_.dynamics_rollout_dim_, this->stream_, false);
    }

    // Copy the costs back to the host
    HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(), this->trajectory_costs_d_,
                                 this->getNumRollouts() * sizeof(float), cudaMemcpyDeviceToHost, this->stream_));

    HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_nominal_.data(), trajectory_costs_nominal_d,
                                 this->getNumRollouts() * sizeof(float), cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));

    this->setBaseline(mppi::kernels::computeBaselineCost(this->trajectory_costs_.data(), this->getNumRollouts()), 0);
    this->setBaseline(
        mppi::kernels::computeBaselineCost(this->trajectory_costs_nominal_.data(), this->getNumRollouts()), 1);

    // Launch the norm exponential kernel for both actual and nominal
    mppi::kernels::launchNormExpKernel(this->getNumRollouts(), this->getNormExpThreads(), this->trajectory_costs_d_,
                                       1.0 / this->getLambda(), this->getBaselineCost(0), this->stream_, false);

    mppi::kernels::launchNormExpKernel(this->getNumRollouts(), this->getNormExpThreads(), trajectory_costs_nominal_d,
                                       1.0 / this->getLambda(), this->getBaselineCost(1), this->stream_, false);

    HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(), this->trajectory_costs_d_,
                                 this->getNumRollouts() * sizeof(float), cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaMemcpyAsync(trajectory_costs_nominal_.data(), trajectory_costs_nominal_d,
                                 this->getNumRollouts() * sizeof(float), cudaMemcpyDeviceToHost, this->stream_));
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));

    // Compute the normalizer
    this->setNormalizer(mppi::kernels::computeNormalizer(this->trajectory_costs_.data(), this->getNumRollouts()), 0);
    this->setNormalizer(
        mppi::kernels::computeNormalizer(this->trajectory_costs_nominal_.data(), this->getNumRollouts()), 1);

    // Compute real free energy
    mppi::kernels::computeFreeEnergy(this->free_energy_statistics_.real_sys.freeEnergyMean,
                                     this->free_energy_statistics_.real_sys.freeEnergyVariance,
                                     this->free_energy_statistics_.real_sys.freeEnergyModifiedVariance,
                                     this->trajectory_costs_.data(), this->getNumRollouts(), this->getBaselineCost(0),
                                     this->getLambda());

    // Compute Nominal State free Energy
    mppi::kernels::computeFreeEnergy(this->free_energy_statistics_.nominal_sys.freeEnergyMean,
                                     this->free_energy_statistics_.nominal_sys.freeEnergyVariance,
                                     this->free_energy_statistics_.nominal_sys.freeEnergyModifiedVariance,
                                     this->trajectory_costs_nominal_.data(), this->getNumRollouts(),
                                     this->getBaselineCost(1), this->getLambda());

    // Compute the cost weighted average
    this->sampler_->updateDistributionParamsFromDevice(this->trajectory_costs_d_, this->getNormalizerCost(0), 0, false);
    this->sampler_->updateDistributionParamsFromDevice(trajectory_costs_nominal_d, this->getNormalizerCost(1), 1,
                                                       false);

    // Transfer the new control to the host
    this->sampler_->setHostOptimalControlSequence(this->control_.data(), 0, false);
    this->sampler_->setHostOptimalControlSequence(this->nominal_control_trajectory_.data(), 1, true);

    // this->logger_->debug("Actual baseline: %f\n", this->getBaselineCost(0));
    // this->logger_->debug("Nominal baseline: %f\n", this->getBaselineCost(1));

    if (this->getBaselineCost(0) < this->getBaselineCost(1) + getNominalThreshold())
    {
      // In this case, the disturbance the made the nominal and actual states differ improved the cost.
      // std::copy(state_trajectory.begin(), state_trajectory.end(), nominal_state_trajectory_.begin());
      // std::copy(control_trajectory.begin(), control_trajectory.end(), nominal_control_.begin());
      this->free_energy_statistics_.nominal_state_used = 0;
      nominal_state_trajectory_ = this->state_;
      nominal_control_trajectory_ = this->control_;
    }
    else
    {
      this->free_energy_statistics_.nominal_state_used = 1;
    }

    // Outside of this loop, we will utilize the nominal state trajectory and the nominal control trajectory to compute
    // the optimal feedback gains using our ancillary controller, then apply feedback inside our main while loop at the
    // same rate as our state estimator.
  }
  smoothControlTrajectory();

  // Compute nominal and real state and output trajectories
  computeStateTrajectory(state);  // Input is the actual state

  this->free_energy_statistics_.real_sys.normalizerPercent = this->getNormalizerCost(0) / this->getNumRollouts();
  this->free_energy_statistics_.real_sys.increase =
      this->getBaselineCost(0) - this->free_energy_statistics_.real_sys.previousBaseline;
  this->free_energy_statistics_.nominal_sys.normalizerPercent = this->getNormalizerCost(1) / this->getNumRollouts();
  this->free_energy_statistics_.nominal_sys.increase =
      this->getBaselineCost(1) - this->free_energy_statistics_.nominal_sys.previousBaseline;

  // Copy back sampled trajectories
  this->copySampledControlFromDevice(false);
  if (this->getKernelChoiceAsEnum() == kernelType::USE_SINGLE_KERNEL)
  {  // copy initial state to vis initial state for use with visualizeKernel
    HANDLE_ERROR(cudaMemcpyAsync(this->vis_initial_state_d_, this->initial_state_d_,
                                 sizeof(float) * DYN_T::STATE_DIM * 2, cudaMemcpyDeviceToDevice, this->vis_stream_));
  }
  this->copyTopControlFromDevice(true);
}

TUBE_MPPI_TEMPLATE
void TubeMPPI::copyControlToDevice(bool synchronize)
{
  this->sampler_->copyImportanceSamplerToDevice(this->control_.data(), 0, false);
  this->sampler_->copyImportanceSamplerToDevice(this->nominal_control_trajectory_.data(), 1, synchronize);
}

TUBE_MPPI_TEMPLATE
void TubeMPPI::allocateCUDAMemory()
{
  PARENT_CLASS::allocateCUDAMemoryHelper(2);
}

TUBE_MPPI_TEMPLATE
void TubeMPPI::slideControlSequence(int steps)
{
  // Propagate the nominal trajectory forward
  updateNominalState(nominal_control_trajectory_.col(0));

  // Save the control history
  this->saveControlHistoryHelper(steps, nominal_control_trajectory_, this->control_history_);

  this->slideControlSequenceHelper(steps, nominal_control_trajectory_);
  this->slideControlSequenceHelper(steps, this->control_);
}

TUBE_MPPI_TEMPLATE
void TubeMPPI::smoothControlTrajectory()
{
  this->smoothControlTrajectoryHelper(nominal_control_trajectory_, this->control_history_);
}

TUBE_MPPI_TEMPLATE
void TubeMPPI::computeStateTrajectory(const Eigen::Ref<const state_array>& x0_actual)
{
  // update the nominal state
  this->computeOutputTrajectoryHelper(nominal_output_trajectory_, nominal_state_trajectory_,
                                      nominal_state_trajectory_.col(0), nominal_control_trajectory_);
  // update the actual state
  this->computeOutputTrajectoryHelper(this->output_, this->state_, x0_actual, this->control_);
}

TUBE_MPPI_TEMPLATE
void TubeMPPI::updateNominalState(const Eigen::Ref<const control_array>& u)
{
  state_array xdot;
  output_array output;
  this->model_->step(nominal_state_trajectory_.col(0), nominal_state_trajectory_.col(0), xdot, u, output, 0,
                     this->getDt());
}

TUBE_MPPI_TEMPLATE
void TubeMPPI::copySampledControlFromDevice(bool synchronize)
{
  // if mem is not inited don't use it
  if (!this->sampled_states_CUDA_mem_init_)
  {
    return;
  }

  int num_sampled_trajectories = this->getNumberSampledTrajectories();
  int real_system_offset = this->getTotalSampledTrajectories();
  this->random_sample_indices_.resize(num_sampled_trajectories);
  if (this->perc_sampled_control_trajectories_ > 0.98)
  {
    // if above threshold just do everything
    std::iota(this->random_sample_indices_.begin(), this->random_sample_indices_.end(), 0);
  }
  else
  {
    // Create sample list without replacement
    this->random_sample_indices_ =
        mppi::math::sample_without_replacement(num_sampled_trajectories, this->getNumRollouts());
  }

  // this explicitly adds the optimized control sequence
  HANDLE_ERROR(cudaMemcpyAsync(this->sampled_outputs_d_, this->output_.data(),
                               sizeof(float) * this->getNumTimesteps() * DYN_T::OUTPUT_DIM, cudaMemcpyHostToDevice,
                               this->vis_stream_));
  HANDLE_ERROR(cudaMemcpyAsync(
      this->sampler_->getVisControlSample(0, 0, 0), this->sampler_->getDeviceOptimalControlSequence(0),
      sizeof(float) * this->getNumTimesteps() * DYN_T::CONTROL_DIM, cudaMemcpyDeviceToDevice, this->vis_stream_));

  for (int i = 1; i < num_sampled_trajectories; i++)
  {
    // Copy Real System
    HANDLE_ERROR(cudaMemcpyAsync(
        this->sampled_outputs_d_ + i * this->getNumTimesteps() * DYN_T::OUTPUT_DIM,
        this->output_d_ + this->random_sample_indices_[i] * this->getNumTimesteps() * DYN_T::OUTPUT_DIM,
        sizeof(float) * this->getNumTimesteps() * DYN_T::OUTPUT_DIM, cudaMemcpyDeviceToDevice, this->vis_stream_));
    HANDLE_ERROR(cudaMemcpyAsync(this->sampler_->getVisControlSample(i, 0, 0),
                                 this->sampler_->getControlSample(this->random_sample_indices_[i], 0, 0),
                                 sizeof(float) * this->getNumTimesteps() * DYN_T::CONTROL_DIM, cudaMemcpyDeviceToDevice,
                                 this->vis_stream_));

    // Copy Nominal System
    HANDLE_ERROR(cudaMemcpyAsync(
        this->sampled_outputs_d_ + (real_system_offset + i) * this->getNumTimesteps() * DYN_T::OUTPUT_DIM,
        this->output_d_ +
            (this->getNumRollouts() + this->random_sample_indices_[i]) * this->getNumTimesteps() * DYN_T::OUTPUT_DIM,
        sizeof(float) * this->getNumTimesteps() * DYN_T::OUTPUT_DIM, cudaMemcpyDeviceToDevice, this->vis_stream_));
    HANDLE_ERROR(cudaMemcpyAsync(this->sampler_->getVisControlSample(i, 0, 1),
                                 this->sampler_->getControlSample(this->random_sample_indices_[i], 0, 1),
                                 sizeof(float) * this->getNumTimesteps() * DYN_T::CONTROL_DIM, cudaMemcpyDeviceToDevice,
                                 this->vis_stream_));
  }
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(this->vis_stream_));
  }
}

TUBE_MPPI_TEMPLATE
void TubeMPPI::copyTopControlFromDevice(bool synchronize)
{
  // if mem is not inited don't use it
  if (!this->sampled_states_CUDA_mem_init_ || this->getNumberTopControlTrajectories() <= 0)
  {
    return;
  }

  // Important note: Highest weighted trajectories are the ones with the lowest cost
  int start_top_control_traj_index = this->getNumberSampledTrajectories();
  nominal_top_n_sample_indices_.resize(this->getNumberTopControlTrajectories());
  this->top_n_sample_indices_.resize(this->getNumberTopControlTrajectories());
  // Start by filling in the top samples list with the first n in the trajectory
  for (int i = 0; i < this->getNumberTopControlTrajectories(); i++)
  {
    nominal_top_n_sample_indices_[i] = i;
    this->top_n_sample_indices_[i] = i;
  }

  // Calculate min weight in the current top samples list
  int nominal_min_sample_index = 0;
  float nominal_min_sample_value = 0;
  int real_min_sample_index = 0;
  float real_min_sample_value = 0;
  std::tie(real_min_sample_index, real_min_sample_value) =
      this->findMinIndexAndValue(this->top_n_sample_indices_, this->trajectory_costs_);
  std::tie(nominal_min_sample_index, nominal_min_sample_value) =
      this->findMinIndexAndValue(nominal_top_n_sample_indices_, trajectory_costs_nominal_);

  // find top n samples by removing the smallest weights from the list
  for (int i = this->getNumberTopControlTrajectories(); i < this->getNumRollouts(); i++)
  {
    if (trajectory_costs_nominal_[i] > nominal_min_sample_value)
    {  // Remove the smallest weight in the current list and add the new index
      nominal_top_n_sample_indices_[nominal_min_sample_index] = i;
      // recalculate min weight in the current list
      std::tie(nominal_min_sample_index, nominal_min_sample_value) =
          this->findMinIndexAndValue(nominal_top_n_sample_indices_, trajectory_costs_nominal_);
    }
    if (this->trajectory_costs_[i] > real_min_sample_value)
    {  // Remove the smallest weight in the current list and add the new index
      this->top_n_sample_indices_[real_min_sample_index] = i;
      // recalculate min weight in the current list
      std::tie(real_min_sample_index, real_min_sample_value) =
          this->findMinIndexAndValue(this->top_n_sample_indices_, this->trajectory_costs_);
    }
  }

  switch (this->params_.copy_type_)
  {
    case PARAMS_T::TOP_COPY_TYPE::COPY_TOP_REAL_SAMPLES_WITH_MATCHING_NOMINAL_SAMPLES:
      // Copy real sample indices to nominal sample list
      nominal_top_n_sample_indices_ = this->top_n_sample_indices_;
      break;
    case PARAMS_T::TOP_COPY_TYPE::COPY_TOP_NOMINAL_SAMPLES_WITH_MATCHING_REAL_SAMPLES:
      // Copy nominal sample indices to real sample list
      this->top_n_sample_indices_ = nominal_top_n_sample_indices_;
      break;
    case PARAMS_T::TOP_COPY_TYPE::COPY_TOP_REAL_AND_NOMINAL_SAMPLES_INDEPENDENTLY:
    default:
      break;
  }

  // Copy top n samples to this->sampled_noise_d_ after the randomly sampled trajectories
  this->top_n_costs_.resize(this->getNumberTopControlTrajectories());
  for (int i = 0; i < this->getNumberTopControlTrajectories(); i++)
  {
    this->top_n_costs_[i] = this->trajectory_costs_[this->top_n_sample_indices_[i]] / this->getNormalizerCost(0);
    // Real system
    HANDLE_ERROR(cudaMemcpyAsync(
        this->sampled_outputs_d_ + (start_top_control_traj_index + i) * this->getNumTimesteps() * DYN_T::OUTPUT_DIM,
        this->output_d_ + this->top_n_sample_indices_[i] * this->getNumTimesteps() * DYN_T::OUTPUT_DIM,
        sizeof(float) * this->getNumTimesteps() * DYN_T::OUTPUT_DIM, cudaMemcpyDeviceToDevice, this->vis_stream_));
    HANDLE_ERROR(cudaMemcpyAsync(this->sampler_->getVisControlSample(start_top_control_traj_index + i, 0, 0),
                                 this->sampler_->getControlSample(this->top_n_sample_indices_[i], 0, 0),
                                 sizeof(float) * this->getNumTimesteps() * DYN_T::CONTROL_DIM, cudaMemcpyDeviceToDevice,
                                 this->vis_stream_));

    // Nominal System
    HANDLE_ERROR(cudaMemcpyAsync(
        this->sampled_outputs_d_ + (2 * start_top_control_traj_index + this->getNumberTopControlTrajectories() + i) *
                                       this->getNumTimesteps() * DYN_T::OUTPUT_DIM,
        this->output_d_ +
            (this->getNumRollouts() + nominal_top_n_sample_indices_[i]) * this->getNumTimesteps() * DYN_T::OUTPUT_DIM,
        sizeof(float) * this->getNumTimesteps() * DYN_T::OUTPUT_DIM, cudaMemcpyDeviceToDevice, this->vis_stream_));
    HANDLE_ERROR(cudaMemcpyAsync(this->sampler_->getVisControlSample(start_top_control_traj_index + i, 0, 1),
                                 this->sampler_->getControlSample(nominal_top_n_sample_indices_[i], 0, 1),
                                 sizeof(float) * this->getNumTimesteps() * DYN_T::CONTROL_DIM, cudaMemcpyDeviceToDevice,
                                 this->vis_stream_));
  }
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(this->vis_stream_));
  }
}

TUBE_MPPI_TEMPLATE
void TubeMPPI::calculateSampledStateTrajectories()
{
  int num_sampled_trajectories = this->getTotalSampledTrajectories();
  // control already copied in compute control, so run kernel
  if (this->getKernelChoiceAsEnum() == kernelType::USE_SPLIT_KERNELS)
  {
    mppi::kernels::launchVisualizeCostKernel<COST_T, SAMPLING_T>(
        this->cost_, this->sampler_, this->getDt(), this->getNumTimesteps(), num_sampled_trajectories,
        this->getLambda(), this->getAlpha(), this->sampled_outputs_d_, this->sampled_crash_status_d_,
        this->sampled_costs_d_, this->params_.cost_rollout_dim_, this->stream_, false);
  }
  else if (this->getKernelChoiceAsEnum() == kernelType::USE_SINGLE_KERNEL)
  {
    mppi::kernels::launchVisualizeKernel<DYN_T, COST_T, SAMPLING_T>(
        this->model_, this->cost_, this->sampler_, this->getDt(), this->getNumTimesteps(), num_sampled_trajectories,
        this->getLambda(), this->getAlpha(), this->vis_initial_state_d_, this->sampled_outputs_d_,
        this->sampled_costs_d_, this->sampled_crash_status_d_, this->params_.visualize_dim_, this->stream_, false);
  }

  // copy back results
  for (int i = 0; i < num_sampled_trajectories; i++)
  {
    // Copy back real system
    HANDLE_ERROR(cudaMemcpyAsync(this->sampled_trajectories_[i].data(),
                                 this->sampled_outputs_d_ + i * this->getNumTimesteps() * DYN_T::OUTPUT_DIM,
                                 this->getNumTimesteps() * DYN_T::OUTPUT_DIM * sizeof(float), cudaMemcpyDeviceToHost,
                                 this->vis_stream_));
    HANDLE_ERROR(
        cudaMemcpyAsync(this->sampled_costs_[i].data(), this->sampled_costs_d_ + (i * (this->getNumTimesteps() + 1)),
                        (this->getNumTimesteps() + 1) * sizeof(float), cudaMemcpyDeviceToHost, this->vis_stream_));
    HANDLE_ERROR(cudaMemcpyAsync(this->sampled_crash_status_[i].data(),
                                 this->sampled_crash_status_d_ + (i * this->getNumTimesteps()),
                                 this->getNumTimesteps() * sizeof(int), cudaMemcpyDeviceToHost, this->vis_stream_));

    // Copy back nominal system
    HANDLE_ERROR(cudaMemcpyAsync(
        this->sampled_trajectories_[num_sampled_trajectories + i].data(),
        this->sampled_outputs_d_ + (num_sampled_trajectories + i) * this->getNumTimesteps() * DYN_T::OUTPUT_DIM,
        this->getNumTimesteps() * DYN_T::OUTPUT_DIM * sizeof(float), cudaMemcpyDeviceToHost, this->vis_stream_));
    HANDLE_ERROR(
        cudaMemcpyAsync(this->sampled_costs_[num_sampled_trajectories + i].data(),
                        this->sampled_costs_d_ + ((num_sampled_trajectories + i) * (this->getNumTimesteps() + 1)),
                        (this->getNumTimesteps() + 1) * sizeof(float), cudaMemcpyDeviceToHost, this->vis_stream_));
    HANDLE_ERROR(
        cudaMemcpyAsync(this->sampled_crash_status_[num_sampled_trajectories + i].data(),
                        this->sampled_crash_status_d_ + ((num_sampled_trajectories + i) * this->getNumTimesteps()),
                        this->getNumTimesteps() * sizeof(int), cudaMemcpyDeviceToHost, this->vis_stream_));
  }
  HANDLE_ERROR(cudaStreamSynchronize(this->vis_stream_));
}
