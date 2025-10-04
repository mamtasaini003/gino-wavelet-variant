import numpy as np
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import os
import warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.models.model import GINO
# from core.models.gino_2d.FNO_wavelet import GINOWithWaveletFNO
# from core.utils.compute_sdf import SDF

warnings.filterwarnings('ignore')			
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'			
tf.get_logger().setLevel('ERROR')



# ---- Toggle ClearML ----
use_clearml = True  # ← Set this to True to enable ClearML

if use_clearml:
    from clearml import Task

## ----------------------------------------------------------------- ##
## ---------------- Set up experiment parameters ------------------- ##
## ----------------------------------------------------------------- ##
#i_data_file = "support_files/stator_dataset.txt"
i_output_path = "output"
i_embed_dim = 64
i_hidden_dim = 128

# ---- Enhanced FNO Parameters ----
i_num_fno_layers = 10
i_fno_modes = 18
i_fno_input_mlp_layers = 3    # 3-layer input MLP
i_fno_latent_mlp_layers = 4   # 4-layer latent MLP
i_fno_output_mlp_layers = 3   # 3-layer output MLP
i_fno_hidden_dim = 256        # Hidden dimension for FNO MLPs

# ---- Training Parameters ----
i_lr = 5e-3
i_decay_steps = 50
i_decay_rate = 0.99
i_epochs = 200

# model_type = "Basic"
model_type = "more_layers"
# model_type = "deep_architecture"

# ---- Batching Parameters ----
i_batch_size = 1  # Adjust this based on your GPU memory dont adjust 
i_shuffle_buffer = 10000  # For shuffling the dataset

i_take_name = "case1 sin+cos input" 

os.makedirs(i_output_path, exist_ok=True)

# data = SDF.get_sdf()
# exit()

hyperparameters = {
            "embed_dim": i_embed_dim,
            "hidden_dim": i_hidden_dim,
            "num_fno_layers": i_num_fno_layers,
            "learning_rate": i_lr,
            "decay_steps": i_decay_steps,
            "decay_rate": i_decay_rate,
            "epochs": i_epochs,
            "batch_size": i_batch_size,
            "wavelet_type": "haar",
            "high_freq_weight": 0.3
        }

with open(f"{i_output_path}/hyperparameters.txt", "w") as f:
    for k, v in hyperparameters.items():
        f.write(f"{k}: {v}\n")       

# ------------------ REPLACEMENT BLOCK START ------------------
# (replace from 'files = [ ... ]' down to creation of train_dataset)
files = [
    ("support_files/donut_poisson_txt/donut_angle_000.txt", (0.0)),
    ("support_files/donut_poisson_txt/donut_angle_036.txt",  (48.0)),
    ("support_files/donut_poisson_txt/donut_angle_072.txt",  (72.0)),
    ("support_files/donut_poisson_txt/donut_angle_108.txt",  (108.0)),
    ("support_files/donut_poisson_txt/donut_angle_144.txt",  (144.0)),
    ("support_files/donut_poisson_txt/donut_angle_180.txt",  (180.0)),
    ("support_files/donut_poisson_txt/donut_angle_216.txt",  (216.0)),
    ("support_files/donut_poisson_txt/donut_angle_252.txt",  (252.0)),
    ("support_files/donut_poisson_txt/donut_angle_288.txt",  (288.0)),
    ("support_files/donut_poisson_txt/donut_angle_324.txt",  (324.0)),
]

# Collect per-file arrays (each element = one file, shape: [n_points, ...])
coords_list = []   # will store [x, y, theta] per file (normed later)
vals_list = []     # will store f (A-field) per file
u_list = []        # target (u) per file

for file_A, angle in files:
    i_data_test_A = np.loadtxt(file_A, skiprows=1)   

    x = i_data_test_A[:, 0] * 1000.0  # mm
    y = i_data_test_A[:, 1] * 1000.0  # mm
    u = i_data_test_A[:, 5]

    # geometric theta from x,y
    theta = np.arctan2(y, x)  

    # sin/cos of given azimuth (metadata angle)
    sin_angles_train = np.full_like(x, np.sin(np.deg2rad(angle)))
    cos_angles_train = np.full_like(x, np.cos(np.deg2rad(angle)))

    # Stack coords as (x, y, theta)
    coords = np.stack([x, y], axis=-1)   # shape (n_points, 3)

    # Stack angle features (sin, cos)
    vals = np.stack([sin_angles_train, cos_angles_train], axis=-1)  # shape (n_points, 2)

    coords_list.append(coords)
    vals_list.append(vals)
    u_list.append(u)

# Convert lists to arrays
coords_arr = np.stack(coords_list, axis=0)    # (n_files, n_points, 3)
vals_arr = np.stack(vals_list, axis=0)        # (n_files, n_points, 2)
u_arr = np.stack(u_list, axis=0)              # (n_files, n_points)

# Flatten for normalization
coords_flat = coords_arr.reshape(-1, coords_arr.shape[-1])  # (n_files*n_points, 3)
x_flat = coords_flat[:, 0]
y_flat = coords_flat[:, 1]

vals_flat = vals_arr.reshape(-1, vals_arr.shape[-1])  # (n_files*n_points, 2)
sin_theta_flat = vals_flat[:, 0]
cos_theta_flat = vals_flat[:, 1]

# mins / maxs
x_min, x_max = x_flat.min(), x_flat.max()
y_min, y_max = y_flat.min(), y_flat.max()
sin_theta_min, sin_theta_max = sin_theta_flat.min(), sin_theta_flat.max()
cos_theta_min, cos_theta_max = cos_theta_flat.min(), cos_theta_flat.max()
u_min, u_max = u_arr.reshape(-1).min(), u_arr.reshape(-1).max()

# avoid division by zero defensively
def norm01(a, a_min, a_max):
    if a_max - a_min == 0:
        return np.zeros_like(a)
    return 2.0 * (a - a_min) / (a_max - a_min) - 1.0  # scaled to [-1,1]

# Apply normalization
x_norm = norm01(coords_arr[..., 0], x_min, x_max)
y_norm = norm01(coords_arr[..., 1], y_min, y_max)
sin_theta_norm = norm01(vals_arr[..., 0], sin_theta_min, sin_theta_max)
cos_theta_norm = norm01(vals_arr[..., 1], cos_theta_min, cos_theta_max)
u_norm = norm01(u_arr, u_min, u_max)            

# Build inputs for model. Important: model API expects coords and vals separately.
# coords -> [x, y, theta] (features dimension=3)
input_pts_train = np.stack([x_norm, y_norm], axis=-1).astype(np.float32)   # (n_files, n_points, 3)
input_vals_train = np.stack([sin_theta_norm, cos_theta_norm], axis=-1).astype(np.float32)    # (n_files, n_points)
u_true_train = u_norm.astype(np.float32)          # (n_files, n_points)

# ---- Test data (example: Az_5) ----
angle_test = (156.0) 
i_data_test_A = np.loadtxt("support_files/donut_poisson_txt/donut_angle_156.txt", skiprows=1)
x_test = i_data_test_A[:,0] * 1000.0
y_test = i_data_test_A[:,1] * 1000.0
u_test = i_data_test_A[:,5] 
sin_theta_test = np.full_like(x_test, np.sin(np.deg2rad(angle_test)))
cos_theta_test = np.full_like(x_test, np.cos(np.deg2rad(angle_test)))

x_test_norm = norm01(x_test, x_min, x_max)
y_test_norm = norm01(y_test, y_min, y_max)
sin_theta_test_norm = norm01(sin_theta_test, sin_theta_min, sin_theta_max)
cos_theta_test_norm = norm01(cos_theta_test, cos_theta_min, cos_theta_max)
u_test_norm = norm01(u_test, u_min, u_max)

input_pts_test = np.stack([x_test_norm, y_test_norm], axis=-1).astype(np.float32)   # (n_points, 3)
input_vals_test = np.stack([sin_theta_test_norm, cos_theta_test_norm], axis=-1).astype(np.float32)   # (n_points, 3)
u_true_test = u_test_norm.astype(np.float32)      # (n_points,)

# Expand test dims to have batch dimension (1, n_points, ...)
test_pts = input_pts_test[None, ...]
test_vals = input_vals_test[None, ...]
test_u = u_true_test[None, ...]

# ---- Create TensorFlow Dataset for Per-Geometry Batching ----
def create_tf_dataset(pts, vals, u, batch_size, shuffle=False):
    """
    pts: (n_geometries, n_points, feature_dim)
    vals: (n_geometries, n_points)
    u: (n_geometries, n_points)
    
    Returns a dataset where each batch contains full geometries.
    """
    dataset = tf.data.Dataset.from_tensor_slices((pts, vals, u))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(pts), seed=42)

    # Each batch contains full geometries (not mixed points)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# Create training dataset
train_dataset = create_tf_dataset(
    input_pts_train.astype(np.float32), 
    input_vals_train.astype(np.float32), 
    u_true_train.astype(np.float32),
    i_batch_size,
    shuffle=True
)

# For evaluation, we can process the full test set at once or in batches
test_pts = input_pts_test[None, ...].astype(np.float32)
test_vals = input_vals_test[None, ...].astype(np.float32)
test_u = u_true_test[None, ...].astype(np.float32)

# ---- Import and Instantiate Model ----
# Import the fixed wavelet-enhanced FNO
model = GINO(
        embed_dim=i_embed_dim,
        hidden_dim=i_hidden_dim,
        num_fno_layers=i_num_fno_layers
    )
print("Created regular GINO model")

# ---- Optimizer and Loss ----
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    i_lr, #initial learning rate
    i_decay_steps,    # how many steps before applying decay
    i_decay_rate,    # multiply LR by this factor
    staircase=False      # if True: step-wise decay; if False: smooth
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
loss_fn = tf.keras.losses.MeanSquaredError()

ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(
    ckpt, directory=i_output_path, max_to_keep=1
)

# ---- Restore latest checkpoint if needed ----
# latest_ckpt = ckpt_manager.latest_checkpoint
latest_ckpt=False
if latest_ckpt:
    ckpt.restore(latest_ckpt)
    print(f"Restored from {latest_ckpt}")
else:
    print("No checkpoint found, initializing from scratch.")

if use_clearml:
    task = Task.init(project_name='GINO_wavelet', task_name=i_take_name)
    task.connect(hyperparameters)
    logger = task.get_logger()
else:
    task = None
    class DummyLogger:
        def report_scalar(self, *args, **kwargs): pass
    logger = DummyLogger()

# ---- Improved Training Function for Batched Data ----
def train_step_single_batch(batch_pts, batch_vals, batch_u):
    with tf.GradientTape() as tape:
        batch_size = tf.shape(batch_pts)[0]
        total_points = tf.shape(batch_pts)[1]

        batch_pts_reshaped = tf.reshape(batch_pts, (batch_size, total_points, -1))
        
        batch_vals_reshaped = tf.reshape(batch_vals, (batch_size, total_points, -1))
        batch_vals_reshaped = tf.ensure_shape(batch_vals_reshaped, [None, None, 2])

        pred = model((batch_pts_reshaped, batch_vals_reshaped, batch_pts_reshaped))
        loss = loss_fn(batch_u, pred)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss



# ---- Training Loop with Improved Batching ----
epochs = i_epochs
train_losses = []
epoch_losses = []

print(f"Training with batch size: {i_batch_size}")
print(f"Total training samples: {len(input_pts_train)}")

for epoch in range(epochs):
    epoch_loss_sum = 0.0
    num_batches = 0
    
    # Process each batch
    for batch_pts, batch_vals, batch_u in train_dataset:
        try:
            # Use the optimized training step
            loss = train_step_single_batch(batch_pts, batch_vals, batch_u)
            
            epoch_loss_sum += loss.numpy()
            num_batches += 1
            
        except tf.errors.ResourceExhaustedError:
            print(f"Out of memory on batch {num_batches}. Consider reducing batch size.")
            break
        except tf.errors.InvalidArgumentError as e:
            print(f"Shape mismatch error on batch {num_batches}: {e}")
            # Try to debug the shapes
            print(f"Batch shapes - pts: {batch_pts.shape}, vals: {batch_vals.shape}, u: {batch_u.shape}")
            break
    
    # Average loss for this epoch
    if num_batches > 0:
        avg_epoch_loss = epoch_loss_sum / num_batches
        epoch_losses.append(avg_epoch_loss)
        
        if (epoch+1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.learning_rate.numpy()
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_epoch_loss:.6f} | LR: {current_lr:.2e} | Batches: {num_batches}")
        
        # Log to ClearML
        if use_clearml and (epoch+1) % 10 == 0:
            logger.report_scalar("training", "loss", iteration=epoch, value=avg_epoch_loss)
            logger.report_scalar("training", "learning_rate", iteration=epoch, value=optimizer.learning_rate.numpy())
    else:
        print(f"Epoch {epoch+1}: No batches processed successfully")
        break

    # Save checkpoint periodically
    if (epoch + 1) % 100 == 0 or (epoch + 1) == epochs:
        save_path = ckpt_manager.save()
        print(f"Checkpoint saved at {save_path}")

# ---- Save Loss Curve ----
if len(epoch_losses) > 0:
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(epoch_losses)+1), epoch_losses, 'b-', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Use log scale for better visualization
    plt.tight_layout()
    plt.savefig(os.path.join(i_output_path, "training_loss.png"), dpi=150)
    plt.show()

# ---- Evaluation ----
print("Evaluating on test set...")
try:
    u_pred_norm = model((test_pts, test_vals, test_pts)).numpy().flatten()
    u_pred = (u_pred_norm + 1) * (u_max - u_min) / 2 + u_min
    # u_test_actual = test_u.flatten() * u_train_max
    u_test_actual = u_test

    mse = np.mean((u_pred - u_test_actual)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(u_pred - u_test_actual))
    l2_norm = np.linalg.norm(u_pred - u_test_actual)
    rel_error = l2_norm / np.linalg.norm(u_test_actual)

    # Calculate R² score
    ss_res = np.sum((u_test_actual - u_pred)**2)
    ss_tot = np.sum((u_test_actual - np.mean(u_test_actual))**2)
    r2_score = 1 - (ss_res / ss_tot)

    # Save detailed error metrics
    with open(os.path.join(i_output_path, "error_metrics.txt"), "w") as f:
        f.write(f"Mean Squared Error (MSE): {mse:.6e}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.6e}\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.6e}\n")
        f.write(f"L2 Norm of Error: {l2_norm:.6e}\n")
        f.write(f"Relative L2 Error: {rel_error:.6e}\n")
        f.write(f"R² Score: {r2_score:.6f}\n")

    print(f"Test MSE: {mse:.6e}")
    print(f"Test RMSE: {rmse:.6e}")
    print(f"Test MAE: {mae:.6e}")
    print(f"L2 Norm of Error: {l2_norm:.6e}")
    print(f"Relative L2 Error: {rel_error:.6e}")
    print(f"R² Score: {r2_score:.6f}")

    # Log metrics to ClearML
    if use_clearml:
        logger.report_scalar("test", "MSE", iteration=0, value=mse)
        logger.report_scalar("test", "RMSE", iteration=0, value=rmse)
        logger.report_scalar("test", "MAE", iteration=0, value=mae)
        logger.report_scalar("test", "L2 Norm", iteration=0, value=l2_norm)
        logger.report_scalar("test", "Relative L2 Error", iteration=0, value=rel_error)
        logger.report_scalar("test", "R2 Score", iteration=0, value=r2_score)

except Exception as e:
    print(f"Evaluation failed: {e}")
    print("This might be due to shape mismatches or memory issues.")
    u_pred = None
    u_test_actual = None

# ---- Visualization (only if evaluation succeeded) ----
if u_pred is not None and u_test_actual is not None:
    
    # Prediction vs True scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(u_test_actual, u_pred, s=3, alpha=0.6, c='blue')
    
    # Perfect prediction line
    min_val = min(u_test_actual.min(), u_pred.min())
    max_val = max(u_test_actual.max(), u_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel("Analytical solution", fontsize=12)
    plt.ylabel("Predicted Solution", fontsize=12)
    plt.title(f"Point-wise absolute error (R² = {r2_score:.4f})", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(i_output_path, "point-wise_abs_error.png"), dpi=150)
    plt.show()

    # True field
    plt.figure(figsize=(10, 8))
    sc1 = plt.scatter(x_test_norm, y_test_norm, c=u_test_actual, cmap='viridis', s=3)
    plt.colorbar(sc1, label='Analytical solution')
    plt.xlabel("x (normalized)", fontsize=12)
    plt.ylabel("y (normalized)", fontsize=12)
    plt.title("Analytical solution", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(i_output_path, "analytical_solution.png"), dpi=150)
    plt.show()

    # Predicted field
    plt.figure(figsize=(10, 8))
    sc2 = plt.scatter(x_test_norm, y_test_norm, c=u_pred, cmap='viridis', s=3)
    plt.colorbar(sc2, label='Predicted solution')
    plt.xlabel("x (normalized)", fontsize=12)
    plt.ylabel("y (normalized)", fontsize=12)
    plt.title("Predicted solution", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(i_output_path, "predicted_solution.png"), dpi=150)
    plt.show()

    # Error field
    error_field = np.abs(u_test_actual - u_pred)
    plt.figure(figsize=(10, 8))
    sc3 = plt.scatter(x_test_norm, y_test_norm, c=error_field, cmap='plasma', s=3)
    plt.colorbar(sc3, label='| Analytical solution - Predicted solution|')
    plt.xlabel("x (normalized)", fontsize=12)
    plt.ylabel("y (normalized)", fontsize=12)
    plt.title("Point-wise Absolute Error field (Test)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(i_output_path, "error_field.png"), dpi=150)
    plt.show()

    # Combined visualization
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    
    sc1 = axs[0].scatter(x_test_norm, y_test_norm, c=u_test_actual, cmap='viridis', s=3)
    axs[0].set_title("Analytical solution", fontsize=14)
    axs[0].set_xlabel("x (normalized)")
    axs[0].set_ylabel("y (normalized)")
    plt.colorbar(sc1, ax=axs[0])

    sc2 = axs[1].scatter(x_test_norm, y_test_norm, c=u_pred, cmap='viridis', s=3)
    axs[1].set_title("Predicted solution", fontsize=14)
    axs[1].set_xlabel("x (normalized)")
    axs[1].set_ylabel("y (normalized)")
    plt.colorbar(sc2, ax=axs[1])

    sc3 = axs[2].scatter(x_test_norm, y_test_norm, c=error_field, cmap='plasma', s=3)
    axs[2].set_title("Point-wise Absolute Error", fontsize=14)
    axs[2].set_xlabel("x (normalized)")
    axs[2].set_ylabel("y (normalized)")
    plt.colorbar(sc3, ax=axs[2])
    plt.suptitle(f"GINO with {model_type} - Test Results", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(i_output_path, "combined_fields.png"), dpi=150)
    plt.show()

    # Save predictions
    np.savetxt(os.path.join(i_output_path, "u_predicted.txt"), u_pred)

# ---- Save predictions, actual values, and error field as .txt files ----
if u_pred is not None and u_test_actual is not None:
    # Combine x, y with field values for better traceability
    pred_data = np.column_stack([x_test_norm, y_test_norm, u_pred])
    actual_data = np.column_stack([x_test_norm, y_test_norm, u_test_actual])
    error_data = np.column_stack([x_test_norm, y_test_norm, error_field])

    np.savetxt(os.path.join(i_output_path, "u_predicted.txt"), pred_data,
               header="x_norm y_norm Predicted_u", fmt="%.6e")

    np.savetxt(os.path.join(i_output_path, "u_actual.txt"), actual_data,
               header="x_norm y_norm Actual_u", fmt="%.6e")

    np.savetxt(os.path.join(i_output_path, "u_error.txt"), error_data,
               header="x_norm y_norm |Actual - Predicted|", fmt="%.6e")

    print("Saved u_predicted.txt, u_actual.txt, and u_error.txt")

# ---- Upload artifacts to ClearML ----
if use_clearml:
    try:
        task.upload_artifact("error_metrics", artifact_object=os.path.join(i_output_path, "error_metrics.txt"))
        if u_pred is not None:
            task.upload_artifact("u_predicted", artifact_object=os.path.join(i_output_path, "u_predicted.txt"))
        if os.path.exists(os.path.join(i_output_path, "training_loss.png")):
            task.upload_artifact("training_loss_plot", artifact_object=os.path.join(i_output_path, "training_loss.png"))
        if os.path.exists(os.path.join(i_output_path, "prediction_vs_true.png")):
            task.upload_artifact("prediction_vs_true", artifact_object=os.path.join(i_output_path, "prediction_vs_true.png"))
            task.upload_artifact("true_u_field", artifact_object=os.path.join(i_output_path, "true_u_field.png"))
            task.upload_artifact("predicted_u_field", artifact_object=os.path.join(i_output_path, "predicted_u_field.png"))
            task.upload_artifact("error_field", artifact_object=os.path.join(i_output_path, "error_field.png"))
            task.upload_artifact("combined_fields", artifact_object=os.path.join(i_output_path, "combined_fields.png"))
        print("Artifacts uploaded to ClearML successfully")
    except Exception as e:
        print(f"Failed to upload artifacts to ClearML: {e}")

print(f"Training completed. Results saved to: {i_output_path}")
