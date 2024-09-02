import torch
from gymnasium import Dict

# This code is part of bigger system, that I haven't code myself,
# here is only the imgep algorithm run, that I did code myself (inspired ofc
# from the work made by INRIA bordeaux)


def execute_imgep_exploration(self, exploration_runs, resume_existing_run=False):
    retry = True

    while retry:
        print("STARTING NEW INITIALIZATION")
        print("Exploration: ")

        if resume_existing_run:
            current_run = len(self.policy_archive)
        else:
            self.policy_archive = []
            self.goal_archive = torch.empty((0,) + self.goal_space.shape)
            current_run = 0

        alive_randoms = 0

        while current_run < exploration_runs:
            policy_params = Dict.fromkeys(["init_state", "update_strategy"])

            if len(self.policy_archive) < self.config.initial_random_runs:
                target = None
                selected_policy = None
                goal_achieved = torch.ones(19)

                policy_params["init_state"] = self.system.init_space.sample()
                policy_params["update_strategy"] = self.system.strategy_space.sample()
                policy_params["update_strategy"].h /= 3

                self.system.reset(
                    initialization_parameters=policy_params["init_state"],
                    update_rule_parameters=policy_params["update_strategy"],
                )

                with torch.no_grad():
                    self.system.random_obstacle(8)
                    self.system.generate_init_state()
                    results = self.system.run()
                    goal_achieved = self.goal_space.map(results)

                is_failed = goal_achieved[0] > 0.9 or goal_achieved[1] < -0.5
                if not is_failed:
                    alive_randoms += 1

                optimization_steps = 0
                distance_to_goal = None

            else:
                if len(self.policy_archive) - self.config.initial_random_runs < 8:
                    target = torch.ones(3) * -10
                    target[0] = 0.065
                    target[2] = (
                        0.19
                        - (len(self.policy_archive) - self.config.initial_random_runs)
                        * 0.06
                    )
                    target[1] = 0
                else:
                    target = self.sample_interesting_goal()

                if len(self.policy_archive) - self.config.initial_random_runs >= 2:
                    print(f"Run {current_run}, optimizing towards goal: ")
                    print("TARGET =", str(target))

                selected_policy_idx = self.find_source_policy(target)
                selected_policy = self.policy_archive[selected_policy_idx]

                if (
                    len(self.policy_archive) - self.config.initial_random_runs < 8
                    or len(self.policy_archive) % 5 == 0
                ):
                    policy_params["init_state"] = deepcopy(
                        selected_policy["init_state"]
                    )
                    policy_params["update_strategy"] = deepcopy(
                        selected_policy["update_strategy"]
                    )
                    self.system.reset(
                        initialization_parameters=policy_params["init_state"],
                        update_rule_parameters=policy_params["update_strategy"],
                    )
                    iterations = self.config.goal_optimizer.steps
                else:
                    iterations = 15
                    mutation_failed = True
                    while mutation_failed:
                        policy_params["init_state"] = self.system.init_space.mutate(
                            selected_policy["init_state"]
                        )
                        policy_params["update_strategy"] = (
                            self.system.strategy_space.mutate(
                                selected_policy["update_strategy"]
                            )
                        )
                        self.system.reset(
                            initialization_parameters=policy_params["init_state"],
                            update_rule_parameters=policy_params["update_strategy"],
                        )
                        with torch.no_grad():
                            self.system.generate_init_state()
                            results = self.system.run()
                            goal_achieved = self.goal_space.map(results)

                        if (
                            results.states[-1, :, :, 0].sum() > 10
                            or goal_achieved[0] > 0.11
                        ):
                            mutation_failed = False

                if (
                    isinstance(self.system, torch.nn.Module)
                    and self.config.goal_optimizer.steps > 0
                ):
                    optimizer_class = eval(
                        f"torch.optim.{self.config.goal_optimizer.name}"
                    )
                    self.goal_optimizer = optimizer_class(
                        [
                            {
                                "params": self.system.init_state.parameters(),
                                **self.config.goal_optimizer.init_cppn.parameters,
                            },
                            {
                                "params": self.system.step.parameters(),
                                **self.config.goal_optimizer.step.parameters,
                            },
                        ],
                        **self.config.goal_optimizer.parameters,
                    )

                    last_failed = False
                    for optimization_steps in range(1, iterations):
                        self.system.random_obstacle(8)
                        self.system.generate_init_state()
                        results = self.system.run()
                        goal_achieved = self.goal_space.map(results)

                        x = torch.arange(self.system.config.SX)
                        y = torch.arange(self.system.config.SY)
                        xx = x.view(-1, 1).repeat(1, self.system.config.SY)
                        yy = y.repeat(self.system.config.SX, 1)
                        X = (
                            xx - (target[1] + 0.5) * self.system.config.SX
                        ).float() / 35
                        Y = (
                            yy - (target[2] + 0.5) * self.system.config.SY
                        ).float() / 35
                        D = torch.sqrt(X**2 + Y**2)
                        mask = 0.85 * (D < 0.5).float() + 0.15 * (D < 1).float()

                        loss = (
                            (0.9 * mask - results.states[-1, :, :, 0])
                            .pow(2)
                            .sum()
                            .sqrt()
                        )

                        self.goal_optimizer.zero_grad()
                        loss.backward()
                        self.goal_optimizer.step()

                        self.system.step.compute_kernel()

                        failed = results.states[-1, :, :, 0].sum() < 10
                        if failed and last_failed:
                            self.goal_optimizer.zero_grad()
                            break
                        last_failed = failed

                    if (
                        len(self.policy_archive) >= self.config.initial_random_runs
                        and len(self.policy_archive) - self.config.initial_random_runs
                        < 2
                    ):
                        if loss > 19.5:
                            break
                        elif (
                            len(self.policy_archive) - self.config.initial_random_runs
                            == 2
                        ):
                            retry = False

                    self.system.update_initialization_parameters()
                    self.system.update_update_rule_parameters()
                    policy_params["init_state"] = self.system.initialization_parameters
                    policy_params["update_strategy"] = (
                        self.system.update_rule_parameters
                    )
                    distance_to_goal = loss.item()

                goal_achieved = torch.zeros(3).cpu()
                with torch.no_grad():
                    for _ in range(20):
                        self.system.random_obstacle(8)
                        self.system.generate_init_state()
                        results = self.system.run()
                        if results.states[-1, :, :, 0].sum() < 10:
                            goal_achieved[0] = 10
                            break
                        goal_achieved = (
                            goal_achieved + self.goal_space.map(results).cpu() / 20
                        )

                if len(self.policy_archive) - self.config.initial_random_runs >= 2:
                    print("reached=", str(goal_achieved))

            goal_achieved = goal_achieved.cpu()
            self.db.add_run_data(
                id=current_run,
                policy_parameters=policy_params,
                observations=results,
                source_policy_idx=selected_policy_idx,
                target_goal=target,
                reached_goal=goal_achieved,
                n_optim_steps_to_reach_goal=optimization_steps,
                dist_to_target=distance_to_goal,
            )

            self.policy_archive.append(policy_params)
            self.goal_archive = torch.cat(
                [
                    self.goal_archive,
                    goal_achieved.reshape(1, -1).to(self.goal_archive.device).detach(),
                ]
            )

            if len(self.policy_archive) >= self.config.initial_random_runs:
                plt.imshow(self.system.init_wall.cpu())
                plt.scatter(
                    (
                        (self.goal_archive[:, 0] < 0.11).float()
                        * (self.goal_archive[:, 2] > -0.5).float()
                        * (self.goal_archive[:, 2] + 0.5)
                        * self.system.config.SY
                    ).cpu(),
                    (
                        (self.goal_archive[:, 0] < 0.11).float()
                        * (self.goal_archive[:, 1] > -0.5).float()
                        * (self.goal_archive[:, 1] + 0.5)
                        * self.system.config.SX
                    ).cpu(),
                )
                plt.show()

            current_run += 1

            if len(self.policy_archive) == self.config.initial_random_runs:
                if alive_randoms < 2:
                    break
                print(current_run)

            if len(self.policy_archive) == exploration_runs - 1:
                retry = False
