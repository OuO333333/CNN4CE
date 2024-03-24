import tensorflow as tf

class OneGateMoE(tf.keras.layers.Layer):
    def __init__(self, num_experts, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.kernel_size = kernel_size

        self.experts = [
            tf.keras.layers.Conv1D(
                filters=256,
                kernel_size=kernel_size,
                padding="same",
                activation="relu",
                input_shape=(1, 32, 256)  # 指定输入形状
            )
            for _ in range(num_experts)
        ]
        self.gate = tf.keras.layers.Dense(units=num_experts)  # 将 units 改为 num_experts 以匹配专家数量

    def call(self, inputs):
        # inputs shape: (batch_size, 1, 32, 256)

        # Step 1: Calculate relevance scores
        gate_logits = self.gate(inputs[:, 0, :, :])  # 获取批次的第一维度以匹配门的输入形状
        relevance_scores = tf.nn.softmax(gate_logits, axis=-1)
        # gate_logits shape: (batch_size, 32, 5)
        # relevance_scores shape: (batch_size, 32, 5)
        print("gate_logits shape = ", gate_logits.shape)
        print("relevance_scores shape = ", relevance_scores.shape)
        
        # Step 2: Distribute inputs to experts
        expert_inputs = tf.stack([
            inputs * relevance_scores[:, i:i+1] for i in range(self.num_experts)
        ], axis=1)

        # Step 3: Process inputs by experts
        expert_outputs = [expert(expert_input) for expert, expert_input in zip(self.experts, expert_inputs)]

        # Step 4: Combine expert outputs
        outputs = tf.add_n(expert_outputs)

        return outputs  # shape: (batch_size, 1, 32, 256)