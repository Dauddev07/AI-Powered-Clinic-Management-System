"""The three specialist agents: symptom_agent, appointment_agent, general_info_agent.
Each is invoked directly by app.services.chat once app.services.orchestrator.router
has classified the current message — never called from within each other."""
