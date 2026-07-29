SRC := src
UV := uv
ITERS := 300
PROMOTE := 0.5
ENT := 0.01
LRFLOOR := 0.1

all: run

install:
	$(UV) sync --extra ai

run: install
	$(UV) run --extra ai python -m $(SRC)

# PPO training / playback (requires the `ai` optional dependency group).
train: install
	$(UV) run --extra ai python -m $(SRC).ai.train --iterations $(ITERS)

# Train with the difficulty-ramp curriculum (trivially-winnable -> full game),
# promoting a stage each time eval win rate hits PROMOTE (make curriculum PROMOTE=0.8).
curriculum: install
	$(UV) run --extra ai python -m $(SRC).ai.train --iterations $(ITERS) \
		--curriculum --promote-win $(PROMOTE) --ent-coef $(ENT) \
		--lr-floor $(LRFLOOR)

# Continue training the last run's checkpoint (src/ai/ppo_ckpt.pt).
resume: install
	$(UV) run --extra ai python -m $(SRC).ai.train --iterations $(ITERS) --resume

# Resume a curriculum run (restores the stage reached from the checkpoint).
curriculum-resume: install
	$(UV) run --extra ai python -m $(SRC).ai.train --iterations $(ITERS) \
		--curriculum --promote-win $(PROMOTE) --ent-coef $(ENT) \
		--lr-floor $(LRFLOOR) --resume

# AlphaZero self-play (PUCT + learned policy/value): a fast net that learns to
# win from the MCTS-strength search. Ramps difficulty like the PPO curriculum.
alphazero: install
	$(UV) run --extra ai python -m $(SRC).ai.az_train --iterations $(ITERS) \
		--curriculum --promote-win $(PROMOTE)

# Resume an AlphaZero run (restores the stage reached from the checkpoint).
alphazero-resume: install
	$(UV) run --extra ai python -m $(SRC).ai.az_train --iterations $(ITERS) \
		--curriculum --promote-win $(PROMOTE) --resume

play: install
	$(UV) run --extra ai python -m $(SRC).ai.play

lint:
	flake8 $(SRC)
	mypy $(SRC)

clean:
	find . -type d -name "__pycache__" | xargs rm -rf || true
	rm -rf .mypy_cache .pytest_cache build dist *.egg-info