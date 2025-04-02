import streamlit as st
import pandas as pd
import numpy as np
from ortools.sat.python import cp_model
import plotly.graph_objects as go
from io import BytesIO
import base64

# ----------------------
# Helper Functions
# ----------------------

def load_data(uploaded_file):
    """Load data from CSV or Excel file with error handling."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Validate and convert columns
        required_columns = ['id', 'length', 'width', 'height', 'weight', 'quantity', 'stackable', 'fragile', 'priority']
        
        # Perform column validation with helpful error messages
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing columns: {', '.join(missing_columns)}")
            return None
        
        # Convert to appropriate types
        df['id'] = df['id'].astype(str)
        df['stackable'] = df['stackable'].map({1: True, 0: False, True: True, False: False})
        df['fragile'] = df['fragile'].map({1: True, 0: False, True: True, False: False})
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# ----------------------
# Optimization Algorithm
# ----------------------

class Item:
    def __init__(self, id, length, width, height, weight, quantity, stackable, fragile, priority):
        self.id = id
        self.length = float(length)
        self.width = float(width)
        self.height = float(height)
        self.weight = float(weight)
        self.quantity = int(quantity)
        self.stackable = bool(stackable)
        self.fragile = bool(fragile)
        self.priority = int(priority)

class Truck:
    def __init__(self, length, width, height, max_weight):
        self.length = float(length)
        self.width = float(width)
        self.height = float(height)
        self.max_weight = float(max_weight)

class LoadingOptimizer:
    def __init__(self, items, truck):
        self.items = items
        self.truck = truck
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self.x_pos = {}
        self.y_pos = {}
        self.z_pos = {}

    def optimize(self):
        self.setup_variables()
        self.add_constraints()
        self.add_priority_constraints()
        self.add_support_constraints()  # Add gravity/support constraints
        
        status = self.solver.Solve(self.model)
        return status, self.get_solution()

    def setup_variables(self):
        for i, item in enumerate(self.items):
            for q in range(item.quantity):
                self.x_pos[(i, q)] = self.model.NewIntVar(0, int(self.truck.length - item.length), f'x_{i}_{q}')
                self.y_pos[(i, q)] = self.model.NewIntVar(0, int(self.truck.width - item.width), f'y_{i}_{q}')
                self.z_pos[(i, q)] = self.model.NewIntVar(0, int(self.truck.height - item.height), f'z_{i}_{q}')

    def add_constraints(self):
        for i, item in enumerate(self.items):
            for q in range(item.quantity):
                self.model.Add(self.x_pos[(i, q)] + int(item.length) <= int(self.truck.length))
                self.model.Add(self.y_pos[(i, q)] + int(item.width) <= int(self.truck.width))
                self.model.Add(self.z_pos[(i, q)] + int(item.height) <= int(self.truck.height))

        total_weight = sum(item.weight * item.quantity for item in self.items)
        self.model.Add(int(total_weight) <= int(self.truck.max_weight))

        for i in range(len(self.items)):
            for q in range(self.items[i].quantity):
                for j in range(i, len(self.items)):
                    for r in range(self.items[j].quantity):
                        if i == j and q == r:
                            continue
                        
                        b1 = self.model.NewBoolVar(f'b1_{i}_{q}_{j}_{r}')
                        b2 = self.model.NewBoolVar(f'b2_{i}_{q}_{j}_{r}')
                        b3 = self.model.NewBoolVar(f'b3_{i}_{q}_{j}_{r}')
                        b4 = self.model.NewBoolVar(f'b4_{i}_{q}_{j}_{r}')
                        b5 = self.model.NewBoolVar(f'b5_{i}_{q}_{j}_{r}')
                        b6 = self.model.NewBoolVar(f'b6_{i}_{q}_{j}_{r}')
                        
                        self.model.Add(self.x_pos[(i, q)] + int(self.items[i].length) <= self.x_pos[(j, r)]).OnlyEnforceIf(b1)
                        self.model.Add(self.x_pos[(j, r)] + int(self.items[j].length) <= self.x_pos[(i, q)]).OnlyEnforceIf(b2)
                        self.model.Add(self.y_pos[(i, q)] + int(self.items[i].width) <= self.y_pos[(j, r)]).OnlyEnforceIf(b3)
                        self.model.Add(self.y_pos[(j, r)] + int(self.items[j].width) <= self.y_pos[(i, q)]).OnlyEnforceIf(b4)
                        self.model.Add(self.z_pos[(i, q)] + int(self.items[i].height) <= self.z_pos[(j, r)]).OnlyEnforceIf(b5)
                        self.model.Add(self.z_pos[(j, r)] + int(self.items[j].height) <= self.z_pos[(i, q)]).OnlyEnforceIf(b6)
                        
                        self.model.AddBoolOr([b1, b2, b3, b4, b5, b6])

    def add_priority_constraints(self):
        for i, item in enumerate(self.items):
            for q in range(item.quantity):
                priority_weight = 100 * (10 - item.priority)
                self.model.Minimize(priority_weight * self.x_pos[(i, q)])

                for j, other_item in enumerate(self.items):
                    if other_item.priority >= item.priority:
                        continue
                    for r in range(other_item.quantity):
                        overlap = self.model.NewBoolVar(f'overlap_{i}_{q}_{j}_{r}')
                        self.model.Add(self.x_pos[(i, q)] >= self.x_pos[(j, r)] + int(self.items[j].length)).OnlyEnforceIf(overlap.Not())
                        self.model.Add(self.x_pos[(j, r)] >= self.x_pos[(i, q)] + int(self.items[i].length)).OnlyEnforceIf(overlap.Not())
                        self.model.Add(self.y_pos[(i, q)] >= self.y_pos[(j, r)] + int(self.items[j].width)).OnlyEnforceIf(overlap.Not())
                        self.model.Add(self.y_pos[(j, r)] >= self.y_pos[(i, q)] + int(self.items[i].width)).OnlyEnforceIf(overlap.Not())
                        self.model.Add(self.x_pos[(j, r)] <= self.x_pos[(i, q)]).OnlyEnforceIf(overlap)

    def add_support_constraints(self):
        """Add constraints to ensure items have support beneath them (no floating items)."""
        for i, item in enumerate(self.items):
            for q in range(item.quantity):
                # Create a boolean variable that will be true if the item is on the floor
                on_floor = self.model.NewBoolVar(f'on_floor_{i}_{q}')
                
                # Item is on the floor if z_pos is 0
                self.model.Add(self.z_pos[(i, q)] == 0).OnlyEnforceIf(on_floor)
                self.model.Add(self.z_pos[(i, q)] > 0).OnlyEnforceIf(on_floor.Not())
                
                # List of support conditions
                supported_conditions = [on_floor]  # Item is supported if it's on the floor
                
                # For items not on the floor, check if they have proper support
                for j, support_item in enumerate(self.items):
                    for r in range(support_item.quantity):
                        if i == j and q == r:
                            continue  # Skip self
                        
                        # Check if this item can be supported by item j,r
                        can_support = True
                        
                        # Don't stack on fragile items
                        if support_item.fragile:
                            can_support = False
                        if not item.fragile and not support_item.fragile and item.stackable and support_item.stackable:
                        # If the item has lower priority number than the support item
                        # (lower number means higher actual priority)
                            if item.priority < support_item.priority:
                                # Add constraint that this item should be on top of the support item
                                can_support = True
                            else:
                                # Otherwise, prevent this arrangement
                                can_support = False
                            
                        # Don't stack heavy items on much lighter ones
                            
                        if can_support:
                            supports = self.model.NewBoolVar(f'supports_{j}_{r}_{i}_{q}')
                            
                            # Create variables for checking overlap in x-y projection
                            x_overlaps = self.model.NewBoolVar(f'x_overlaps_{j}_{r}_{i}_{q}')
                            y_overlaps = self.model.NewBoolVar(f'y_overlaps_{j}_{r}_{i}_{q}')
                            z_correct = self.model.NewBoolVar(f'z_correct_{j}_{r}_{i}_{q}')
                            
                            # X-axis overlap condition
                            x_begin_correct = self.model.NewBoolVar(f'x_begin_{j}_{r}_{i}_{q}')
                            x_end_correct = self.model.NewBoolVar(f'x_end_{j}_{r}_{i}_{q}')
                            
                            # Item j,r's start is before item i,q's end
                            self.model.Add(self.x_pos[(j, r)] < self.x_pos[(i, q)] + int(item.length)).OnlyEnforceIf(x_begin_correct)
                            self.model.Add(self.x_pos[(j, r)] >= self.x_pos[(i, q)] + int(item.length)).OnlyEnforceIf(x_begin_correct.Not())
                            
                            # Item i,q's start is before item j,r's end
                            self.model.Add(self.x_pos[(i, q)] < self.x_pos[(j, r)] + int(support_item.length)).OnlyEnforceIf(x_end_correct)
                            self.model.Add(self.x_pos[(i, q)] >= self.x_pos[(j, r)] + int(support_item.length)).OnlyEnforceIf(x_end_correct.Not())
                            
                            # Both conditions must be true for x-overlap
                            self.model.AddBoolAnd([x_begin_correct, x_end_correct]).OnlyEnforceIf(x_overlaps)
                            self.model.AddBoolOr([x_begin_correct.Not(), x_end_correct.Not()]).OnlyEnforceIf(x_overlaps.Not())
                            
                            # Y-axis overlap condition
                            y_begin_correct = self.model.NewBoolVar(f'y_begin_{j}_{r}_{i}_{q}')
                            y_end_correct = self.model.NewBoolVar(f'y_end_{j}_{r}_{i}_{q}')
                            
                            # Item j,r's start is before item i,q's end
                            self.model.Add(self.y_pos[(j, r)] < self.y_pos[(i, q)] + int(item.width)).OnlyEnforceIf(y_begin_correct)
                            self.model.Add(self.y_pos[(j, r)] >= self.y_pos[(i, q)] + int(item.width)).OnlyEnforceIf(y_begin_correct.Not())
                            
                            # Item i,q's start is before item j,r's end
                            self.model.Add(self.y_pos[(i, q)] < self.y_pos[(j, r)] + int(support_item.width)).OnlyEnforceIf(y_end_correct)
                            self.model.Add(self.y_pos[(i, q)] >= self.y_pos[(j, r)] + int(support_item.width)).OnlyEnforceIf(y_end_correct.Not())
                            
                            # Both conditions must be true for y-overlap
                            self.model.AddBoolAnd([y_begin_correct, y_end_correct]).OnlyEnforceIf(y_overlaps)
                            self.model.AddBoolOr([y_begin_correct.Not(), y_end_correct.Not()]).OnlyEnforceIf(y_overlaps.Not())
                            
                            # Z-position: item i,q must be directly on top of item j,r
                            self.model.Add(self.z_pos[(i, q)] == self.z_pos[(j, r)] + int(support_item.height)).OnlyEnforceIf(z_correct)
                            self.model.Add(self.z_pos[(i, q)] != self.z_pos[(j, r)] + int(support_item.height)).OnlyEnforceIf(z_correct.Not())
                            
                            # Support condition: all three axes conditions must be met
                            self.model.AddBoolAnd([x_overlaps, y_overlaps, z_correct]).OnlyEnforceIf(supports)
                            self.model.AddBoolOr([x_overlaps.Not(), y_overlaps.Not(), z_correct.Not()]).OnlyEnforceIf(supports.Not())
                            
                            # Calculate minimum support area required (at least 50% of item's base)
                            # For simplified calculation, we approximate with significant overlap
                            significant_x_overlap = self.model.NewBoolVar(f'sig_x_overlap_{j}_{r}_{i}_{q}')
                            significant_y_overlap = self.model.NewBoolVar(f'sig_y_overlap_{j}_{r}_{i}_{q}')
                            
                            # Create auxiliary variables for overlap calculation
                            x_overlap_start = self.model.NewIntVar(0, int(self.truck.length), f'x_overlap_start_{j}_{r}_{i}_{q}')
                            x_overlap_end = self.model.NewIntVar(0, int(self.truck.length), f'x_overlap_end_{j}_{r}_{i}_{q}')
                            y_overlap_start = self.model.NewIntVar(0, int(self.truck.width), f'y_overlap_start_{j}_{r}_{i}_{q}')
                            y_overlap_end = self.model.NewIntVar(0, int(self.truck.width), f'y_overlap_end_{j}_{r}_{i}_{q}')
                            
                            # Determine overlap bounds
                            # Max of start positions
                            self.model.AddMaxEquality(x_overlap_start, [self.x_pos[(i, q)], self.x_pos[(j, r)]])
                            # Min of end positions
                            self.model.AddMinEquality(
                                x_overlap_end, 
                                [self.model.NewIntVar(0, int(self.truck.length), f'x_end_i_{i}_{q}') + self.x_pos[(i, q)], 
                                 self.model.NewIntVar(0, int(self.truck.length), f'x_end_j_{j}_{r}') + self.x_pos[(j, r)]]
                            )
                            
                            # Same for y-axis
                            self.model.AddMaxEquality(y_overlap_start, [self.y_pos[(i, q)], self.y_pos[(j, r)]])
                            self.model.AddMinEquality(
                                y_overlap_end, 
                                [self.model.NewIntVar(0, int(self.truck.width), f'y_end_i_{i}_{q}') + self.y_pos[(i, q)], 
                                 self.model.NewIntVar(0, int(self.truck.width), f'y_end_j_{j}_{r}') + self.y_pos[(j, r)]]
                            )
                            
                            # Determine if overlap is significant (>50%)
                            x_threshold = int(item.length * 1)
                            y_threshold = int(item.width * 1)
                            
                            x_overlap_amount = self.model.NewIntVar(0, int(self.truck.length), f'x_overlap_amount_{j}_{r}_{i}_{q}')
                            y_overlap_amount = self.model.NewIntVar(0, int(self.truck.width), f'y_overlap_amount_{j}_{r}_{i}_{q}')
                            
                            # Calculate overlap amount
                            self.model.Add(x_overlap_amount == x_overlap_end - x_overlap_start).OnlyEnforceIf(x_overlaps)
                            self.model.Add(x_overlap_amount == 0).OnlyEnforceIf(x_overlaps.Not())
                            
                            self.model.Add(y_overlap_amount == y_overlap_end - y_overlap_start).OnlyEnforceIf(y_overlaps)
                            self.model.Add(y_overlap_amount == 0).OnlyEnforceIf(y_overlaps.Not())
                            
                            # Check if overlap meets threshold
                            x_overlap_sufficient = self.model.NewBoolVar(f'x_sufficient_{j}_{r}_{i}_{q}')
                            y_overlap_sufficient = self.model.NewBoolVar(f'y_sufficient_{j}_{r}_{i}_{q}')
                            
                            self.model.Add(x_overlap_amount >= x_threshold).OnlyEnforceIf(x_overlap_sufficient)
                            self.model.Add(x_overlap_amount < x_threshold).OnlyEnforceIf(x_overlap_sufficient.Not())
                            
                            self.model.Add(y_overlap_amount >= y_threshold).OnlyEnforceIf(y_overlap_sufficient)
                            self.model.Add(y_overlap_amount < y_threshold).OnlyEnforceIf(y_overlap_sufficient.Not())
                            
                            # Add support condition if overlap is sufficient
                            good_support = self.model.NewBoolVar(f'good_support_{j}_{r}_{i}_{q}')
                            self.model.AddBoolAnd([supports, x_overlap_sufficient, y_overlap_sufficient]).OnlyEnforceIf(good_support)
                            self.model.AddBoolOr([supports.Not(), x_overlap_sufficient.Not(), y_overlap_sufficient.Not()]).OnlyEnforceIf(good_support.Not())
                            
                            # Add to list of possible supports
                            supported_conditions.append(good_support)
                
                # Each item must have at least one valid support condition
                self.model.AddBoolOr(supported_conditions)

    def get_solution(self):
        solution = {
            'items': [],
            'total_weight': sum(item.weight * item.quantity for item in self.items),
            'total_volume': sum(item.length * item.width * item.height * item.quantity for item in self.items),
            'truck_volume': self.truck.length * self.truck.width * self.truck.height,
            'status': self.solver.StatusName()
        }
        
        for i, item in enumerate(self.items):
            for q in range(item.quantity):
                solution['items'].append({
                    'id': item.id,
                    'priority': item.priority,
                    'x': float(self.solver.Value(self.x_pos[(i, q)])),
                    'y': float(self.solver.Value(self.y_pos[(i, q)])),
                    'z': float(self.solver.Value(self.z_pos[(i, q)])),
                    'length': float(item.length),
                    'width': float(item.width),
                    'height': float(item.height),
                    'weight': float(item.weight),
                    'stackable': bool(item.stackable),
                    'fragile': bool(item.fragile)
                })

        return solution

# ------------------
# Visualization Functions
# ------------------

def create_3d_box_visualization(truck, items, show_axes=True):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{'type': 'scene'}]],
        subplot_titles=["Truck Loading Visualization"]
    )
    
    # Add trailer and truck cab
    def add_truck_body():
        # Trailer flatbed/floor (slightly thicker)
        floor_height = 0.1  # Floor thickness
        
        # Trailer bed as a solid surface (positioned with x=0 at front, length at back)
        fig.add_trace(go.Mesh3d(
            x=[0, truck['length'], truck['length'], 0, 0, truck['length'], truck['length'], 0],
            y=[0, 0, truck['width'], truck['width'], 0, 0, truck['width'], truck['width']],
            z=[-floor_height, -floor_height, -floor_height, -floor_height, 0, 0, 0, 0],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color='#888888',  # Gray color for floor
            opacity=1.0,
            flatshading=True,
            name="Trailer Bed",
            showlegend=False
        ))
        
        # Add trailer frame for better definition
        trailer_lines = [
            # Bottom frame
            [0, 0, 0], [truck['length'], 0, 0],
            [truck['length'], 0, 0], [truck['length'], truck['width'], 0],
            [truck['length'], truck['width'], 0], [0, truck['width'], 0],
            [0, truck['width'], 0], [0, 0, 0],
        ]
        
        x_lines, y_lines, z_lines = zip(*trailer_lines)
        
        fig.add_trace(go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode='lines',
            line=dict(color='#000000', width=3),
            name='Trailer Frame',
            hoverinfo='name',
            showlegend=False
        ))
        
        # Add wheel assemblies to the trailer
        wheel_radius = 0.3
        wheel_width = 0.2
        
        # Wheel positions (multiple axles)
        wheel_positions = [
            # Left side wheels (y=0)
            {'x': truck['length'] * 0.1, 'y': -wheel_width/2, 'z': -wheel_radius},
            {'x': truck['length'] * 0.25, 'y': -wheel_width/2, 'z': -wheel_radius},
            {'x': truck['length'] * 0.6, 'y': -wheel_width/2, 'z': -wheel_radius},
            {'x': truck['length'] * 0.75, 'y': -wheel_width/2, 'z': -wheel_radius},
            
            # Right side wheels (y=width)
            {'x': truck['length'] * 0.1, 'y': truck['width'] + wheel_width/2, 'z': -wheel_radius},
            {'x': truck['length'] * 0.25, 'y': truck['width'] + wheel_width/2, 'z': -wheel_radius},
            {'x': truck['length'] * 0.6, 'y': truck['width'] + wheel_width/2, 'z': -wheel_radius},
            {'x': truck['length'] * 0.75, 'y': truck['width'] + wheel_width/2, 'z': -wheel_radius},
        ]
        
        # Add wheels
        for wheel in wheel_positions:
            # Create wheel as a cylinder approximation with rings
            theta = np.linspace(0, 2*np.pi, 20)
            x_ring = wheel['x'] + np.zeros(20)
            y_ring = wheel['y'] + wheel_radius * np.cos(theta)
            z_ring = wheel['z'] + wheel_radius * np.sin(theta)
            
            fig.add_trace(go.Scatter3d(
                x=x_ring,
                y=y_ring,
                z=z_ring,
                mode='lines',
                line=dict(color='black', width=5),
                showlegend=False
            ))
        
        # Add truck cab (blue as in the image)
        cab_length = truck['length'] * 0.3  # Cab length
        cab_height = truck['height'] * 1.2  # Cab is taller than trailer
        cab_width = truck['width']
        
        # Position the cab at the end of the trailer (x = truck['length'])
        cab_x_start = truck['length']  # Cab extends from truck['length'] to truck['length'] + cab_length
        
        # Cab body
        cab_corners = [
            # Bottom points
            [cab_x_start, 0, 0],
            [cab_x_start + cab_length, 0, 0],
            [cab_x_start + cab_length, cab_width, 0],
            [cab_x_start, cab_width, 0],
            # Top points
            [cab_x_start, 0, cab_height * 0.7],  # Front top (lower)
            [cab_x_start + cab_length, 0, cab_height],  # Back top (higher)
            [cab_x_start + cab_length, cab_width, cab_height],  # Back top (higher)
            [cab_x_start, cab_width, cab_height * 0.7]  # Front top (lower)
        ]
        
        # Create vertices and faces for the cab mesh
        x = [p[0] for p in cab_corners]
        y = [p[1] for p in cab_corners]
        z = [p[2] for p in cab_corners]
        
        i_indices = [0, 0, 0, 1, 4, 4, 7, 7, 4, 0, 3, 2]
        j_indices = [1, 4, 3, 2, 5, 7, 6, 3, 0, 1, 7, 6]
        k_indices = [2, 5, 7, 6, 6, 6, 5, 2, 3, 5, 4, 5]
        
        fig.add_trace(go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i_indices,
            j=j_indices,
            k=k_indices,
            color='#0047AB',  # Blue color for the cab
            opacity=1.0,
            flatshading=True,
            name="Truck Cab",
            showlegend=False
        ))
        
        # Add windshield (sloped)
        fig.add_trace(go.Mesh3d(
            x=[cab_x_start, cab_x_start + cab_length, cab_x_start + cab_length, cab_x_start],
            y=[0, 0, cab_width, cab_width],
            z=[cab_height * 0.7, cab_height, cab_height, cab_height * 0.7],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color='#87CEFA',  # Light blue for windshield
            opacity=0.7,
            flatshading=True,
            name="Windshield",
            showlegend=False
        ))
        
        # Add cab wheels (positioned correctly relative to cab)
        cab_wheel_positions = [
            # Left front wheel
            {'x': cab_x_start + cab_length*0.8, 'y': -wheel_width/2, 'z': -wheel_radius},
            # Right front wheel
            {'x': cab_x_start + cab_length*0.8, 'y': cab_width + wheel_width/2, 'z': -wheel_radius},
            # Left rear wheel
            {'x': cab_x_start + cab_length*0.2, 'y': -wheel_width/2, 'z': -wheel_radius},
            # Right rear wheel
            {'x': cab_x_start + cab_length*0.2, 'y': cab_width + wheel_width/2, 'z': -wheel_radius}
        ]
        
        for wheel in cab_wheel_positions:
            theta = np.linspace(0, 2*np.pi, 20)
            x_ring = wheel['x'] + np.zeros(20)
            y_ring = wheel['y'] + wheel_radius * np.cos(theta)
            z_ring = wheel['z'] + wheel_radius * np.sin(theta)
            
            fig.add_trace(go.Scatter3d(
                x=x_ring,
                y=y_ring,
                z=z_ring,
                mode='lines',
                line=dict(color='black', width=5),
                showlegend=False
            ))
    
    add_truck_body()
    
    # Calculate volume utilization
    def calculate_volume_utilization():
        truck_volume = truck['length'] * truck['width'] * truck['height']
        used_volume = sum(item['length'] * item['width'] * item['height'] for item in items)
        return (used_volume / truck_volume) * 100 if truck_volume > 0 else 0
    
    volume_utilization = calculate_volume_utilization()
    
    # Define colors for container types/priorities (matching image colors)
    priority_styles = {
        0: {'color': '#FF4500', 'name': 'Urgent'},         # Red-orange
        1: {'color': '#FF0000', 'name': 'Very High'},      # Red
        2: {'color': '#FFA500', 'name': 'High'},           # Orange
        3: {'color': '#FFFF00', 'name': 'Medium-High'},    # Yellow
        4: {'color': '#00FF00', 'name': 'Medium'},         # Green
        5: {'color': '#00BFFF', 'name': 'Medium-Low'},     # Light blue
        6: {'color': '#0000FF', 'name': 'Low'},            # Blue
        7: {'color': '#800080', 'name': 'Very Low'},       # Purple
    }
    
    # Add items as shipping containers
    for i, item in enumerate(items):
        # Determine color based on priority
        priority = min(max(item.get('priority', 0), 0), 7)  # Clamp between 0-7
        style = priority_styles.get(priority, priority_styles[0])
        color = style['color']
        
        # Create container corners (positioned relative to x=0 at door/back of cab)
        x = [item['x'], item['x'] + item['length'], item['x'] + item['length'], item['x'], 
             item['x'], item['x'] + item['length'], item['x'] + item['length'], item['x']]
        y = [item['y'], item['y'], item['y'] + item['width'], item['y'] + item['width'], 
             item['y'], item['y'], item['y'] + item['width'], item['y'] + item['width']]
        z = [item['z'], item['z'], item['z'], item['z'], 
             item['z'] + item['height'], item['z'] + item['height'], item['z'] + item['height'], item['z'] + item['height']]
        
        # Define the 6 faces of the container
        i_indices = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
        j_indices = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
        k_indices = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
        
        # Get any custom name or default to the item ID
        item_name = item.get('name', f"Item {item.get('id', i+1)}")
        item_id = item.get('id', f"ID-{i+1}")
        
        hover_text = f"{item_name}<br>ID: {item_id}<br>Priority: {style['name']}<br>Size: {item['length']}x{item['width']}x{item['height']}"
        
        # Add the box as a shipping container
        fig.add_trace(go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i_indices,
            j=j_indices,
            k=k_indices,
            color=color,
            opacity=1.0,  # Solid appearance
            flatshading=True,
            name=f"{item_name} (P:{priority})",
            hovertemplate=hover_text,
            showlegend=True
        ))
        
        # Add wireframe edges for container
        edge_lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
        
        edge_x, edge_y, edge_z = [], [], []
        for line in edge_lines:
            edge_x.extend([x[line[0]], x[line[1]], None])
            edge_y.extend([y[line[0]], y[line[1]], None])
            edge_z.extend([z[line[0]], z[line[1]], None])
        
        fig.add_trace(go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False,
            hoverinfo='none'
        ))
        
        # Add container ridges (horizontal lines) similar to shipping containers
        for h_offset in np.linspace(0, item['height'], 4):
            if h_offset > 0 and h_offset < item['height']:  # Skip top and bottom
                # Side ridges
                for side_y in [item['y'], item['y'] + item['width']]:
                    ridge_x = [item['x'], item['x'] + item['length']]
                    ridge_y = [side_y, side_y]
                    ridge_z = [item['z'] + h_offset, item['z'] + h_offset]
                    
                    fig.add_trace(go.Scatter3d(
                        x=ridge_x,
                        y=ridge_y,
                        z=ridge_z,
                        mode='lines',
                        line=dict(color='black', width=1),
                        showlegend=False,
                        hoverinfo='none'
                    ))
                
                # Front/back ridges
                for side_x in [item['x'], item['x'] + item['length']]:
                    ridge_x = [side_x, side_x]
                    ridge_y = [item['y'], item['y'] + item['width']]
                    ridge_z = [item['z'] + h_offset, item['z'] + h_offset]
                    
                    fig.add_trace(go.Scatter3d(
                        x=ridge_x,
                        y=ridge_y,
                        z=ridge_z,
                        mode='lines',
                        line=dict(color='black', width=1),
                        showlegend=False,
                        hoverinfo='none'
                    ))
        
        # Add text labels similar to shipping container ID codes
        # Position the label on front face of container
        label_x = item['x'] + item['length'] * 0.5
        label_y = item['y'] + 0.01  # Slightly in front
        label_z = item['z'] + item['height'] * 0.5
        
        # Create container ID text
        container_id = f"ID{item_id}-P{priority}"
        
        fig.add_trace(go.Scatter3d(
            x=[label_x],
            y=[label_y],
            z=[label_z],
            mode='text',
            text=[container_id],
            textposition="middle center",
            textfont=dict(color='black', size=10, family='Arial Black'),
            showlegend=False
        ))
    
    # Add loading statistics annotation
    fig.add_annotation(
        text=f"<b>Loading Statistics</b><br>Items: {len(items)}<br>Volume Used: {volume_utilization:.1f}%",
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        font=dict(size=12)
    )
    
    # Create a legend for priority levels
    for priority, style in priority_styles.items():
        if any(item.get('priority', 0) == priority for item in items):
            continue  # Skip if this priority already has a container in the legend
            
        # Add invisible marker just for the legend
        fig.add_trace(go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode='markers',
            marker=dict(color=style['color'], size=10),
            name=f"Priority {priority}: {style['name']}",
            showlegend=True
        ))
    
    # Set layout with improved camera angle for better view
    # Update layout with new axis ranges
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='Length (x=0 at front)',
                range=[-truck['length']*0.1, truck['length']*1.4],
                showbackground=True if show_axes else False,
                visible=show_axes
            ),
            yaxis=dict(
                title='Width',
                range=[-truck['width']*0.15, truck['width']*1.15],
                showbackground=True if show_axes else False,
                visible=show_axes
            ),
            zaxis=dict(
                title='Height',
                range=[-0.5, truck['height']*1.5],
                showbackground=True if show_axes else False,
                visible=show_axes
            ),
            aspectmode='manual',
            aspectratio=dict(
                x=1.5, 
                y=max(0.5, min(1.5, truck['width']/truck['length'])), 
                z=max(0.5, min(1.0, truck['height']/truck['length']))
            ),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.8)
            )
        ),
        title=dict(
            text="3D Truck Loading Visualization",
            font=dict(size=20)
        ),
        margin=dict(r=20, l=20, b=20, t=50),
        # BÜYÜK BOYUT AYARLARI:
        width=1200,  # Genişlik piksel cinsinden (varsayılan 700)
        height=800,  # Yükseklik piksel cinsinden (varsayılan 450)
        autosize=False,  # Otomatik boyutlandırmayı kapat
        legend=dict(
            title="Items by Priority",
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            itemsizing="constant"
        ),  
        template="plotly_white"
    )
    
    return fig


def get_excel_download_link(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Items')
        workbook = writer.book
        worksheet = writer.sheets['Items']
        
        # Add formatting
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#4472C4',
            'font_color': 'white',
            'border': 1})
        
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Auto-adjust columns' width
        for column in df:
            column_length = max(df[column].astype(str).map(len).max(), len(column))
            col_idx = df.columns.get_loc(column)
            worksheet.set_column(col_idx, col_idx, column_length + 2)
    
    processed_data = output.getvalue()
    b64 = base64.b64encode(processed_data).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="truck_loading_template.xlsx">Download Excel Template</a>'

def main():
    st.set_page_config(page_title="Truck Loading Optimizer", page_icon="🚛", layout="wide")
    
    # Initialize session state
    if 'items' not in st.session_state:
        st.session_state['items'] = [
            {'id': 1, 'length': 3.0, 'width': 2.0, 'height': 1.0, 'weight': 50.0, 'quantity': 2, 
             'stackable': True, 'fragile': False, 'priority': 0},
            {'id': 2, 'length': 2.0, 'width': 2.0, 'height': 2.0, 'weight': 30.0, 'quantity': 1, 
             'stackable': False, 'fragile': True, 'priority': 1}
        ]
        
    if 'truck' not in st.session_state:
        st.session_state['truck'] = {
            'length': 10.0,
            'width': 5.0,
            'height': 5.0,
            'max_weight': 5000.0
        }
        
    if 'solution' not in st.session_state:
        st.session_state['solution'] = None
    
    # Title and description
    st.title("🚛 Truck Loading Optimization")
    st.markdown("""
    Optimize how items are loaded into a truck considering dimensions, weight, and priority.
    Items with lower priority numbers (0 = highest) will be placed toward the front of the truck.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("Truck Specifications")
        st.session_state['truck']['length'] = st.number_input("Length (m)", min_value=0.1, value=st.session_state['truck']['length'], step=0.1, format="%.1f")
        st.session_state['truck']['width'] = st.number_input("Width (m)", min_value=0.1, value=st.session_state['truck']['width'], step=0.1, format="%.1f")
        st.session_state['truck']['height'] = st.number_input("Height (m)", min_value=0.1, value=st.session_state['truck']['height'], step=0.1, format="%.1f")
        st.session_state['truck']['max_weight'] = st.number_input("Max Weight (kg)", min_value=0.1, value=st.session_state['truck']['max_weight'], step=1.0, format="%.1f")
        
        st.header("Data Management")
        st.markdown("### Excel Template")
        sample_df = pd.DataFrame([
            {'id': 1, 'length': 3.0, 'width': 2.0, 'height': 1.0, 'weight': 50.0, 'quantity': 2, 
             'stackable': 1, 'fragile': 0, 'priority': 0},
            {'id': 2, 'length': 2.0, 'width': 2.0, 'height': 2.0, 'weight': 30.0, 'quantity': 1, 
             'stackable': 0, 'fragile': 1, 'priority': 1}
        ])
        st.markdown(get_excel_download_link(sample_df), unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload Excel or CSV File", type=['xlsx', 'xls', 'csv'])
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                # Clear existing items when uploading new file
                st.session_state['items'] = []
                
                # Process each row and add to items
                for _, row in df.iterrows():
                    try:
                        new_item = {
                            'id': int(row['id']),
                            'length': float(row['length']),
                            'width': float(row['width']),
                            'height': float(row['height']),
                            'weight': float(row['weight']),
                            'quantity': int(row['quantity']),
                            'stackable': bool(row['stackable']),
                            'fragile': bool(row['fragile']),
                            'priority': int(row['priority'])
                        }
                        st.session_state['items'].append(new_item)
                    except Exception as e:
                        st.error(f"Error processing row {_+2}: {str(e)}")
                        continue
                
                st.success(f"Successfully loaded {len(df)} items from file!")
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Items Configuration")
        
        if st.button("➕ Add New Item", use_container_width=True):
            new_id = max([item['id'] for item in st.session_state['items']], default=0) + 1
            st.session_state['items'].append({
                'id': new_id,
                'length': 1.0,
                'width': 1.0,
                'height': 1.0,
                'weight': 10.0,
                'quantity': 1,
                'stackable': True,
                'fragile': False,
                'priority': 0
            })
        
        items_to_remove = []
        for i, item in enumerate(st.session_state['items']):
            with st.expander(f"Item {item['id']} (Priority: {item['priority']})", expanded=True):
                cols = st.columns(2)
                with cols[0]:
                    st.session_state['items'][i]['id'] = st.number_input("ID", min_value=1, value=item['id'], step=1, key=f"id_{i}")
                    st.session_state['items'][i]['length'] = st.number_input("Length (m)", min_value=0.1, value=item['length'], step=0.1, format="%.1f", key=f"length_{i}")
                    st.session_state['items'][i]['width'] = st.number_input("Width (m)", min_value=0.1, value=item['width'], step=0.1, format="%.1f", key=f"width_{i}")
                    st.session_state['items'][i]['height'] = st.number_input("Height (m)", min_value=0.1, value=item['height'], step=0.1, format="%.1f", key=f"height_{i}")
                with cols[1]:
                    st.session_state['items'][i]['weight'] = st.number_input("Weight (kg)", min_value=0.1, value=item['weight'], step=0.1, format="%.1f", key=f"weight_{i}")
                    st.session_state['items'][i]['quantity'] = st.number_input("Quantity", min_value=1, value=item['quantity'], step=1, key=f"quantity_{i}")
                    st.session_state['items'][i]['priority'] = st.number_input("Priority (0=highest)", min_value=0, max_value=10, value=item['priority'], step=1, key=f"priority_{i}")
                    st.session_state['items'][i]['stackable'] = st.checkbox("Stackable", value=item['stackable'], key=f"stackable_{i}")
                    st.session_state['items'][i]['fragile'] = st.checkbox("Fragile", value=item['fragile'], key=f"fragile_{i}")
                
                if st.button("❌ Remove Item", key=f"remove_{i}"):
                    items_to_remove.append(i)
        
        # Remove items after iteration
        for i in sorted(items_to_remove, reverse=True):
            st.session_state['items'].pop(i)
        
        if st.button("🚀 Optimize Loading", use_container_width=True, type="primary"):
            with st.spinner("Calculating optimal loading configuration..."):
                try:
                    truck_obj = Truck(
                        st.session_state['truck']['length'],
                        st.session_state['truck']['width'],
                        st.session_state['truck']['height'],
                        st.session_state['truck']['max_weight']
                    )
                    
                    items_obj = []
                    for item in st.session_state['items']:
                        items_obj.append(Item(
                            item['id'],
                            item['length'],
                            item['width'],
                            item['height'],
                            item['weight'],
                            item['quantity'],
                            item['stackable'],
                            item['fragile'],
                            item['priority']
                        ))
                    
                    optimizer = LoadingOptimizer(items_obj, truck_obj)
                    status, solution = optimizer.optimize()
                    
                    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                        st.session_state['solution'] = solution
                        st.success("Optimization completed successfully!")
                    else:
                        st.error("No feasible solution found. Check your constraints (dimensions, weight limits).")
                except Exception as e:
                    st.error(f"Optimization failed: {str(e)}")
    
    with col2:
        if st.session_state['solution']:
            st.header("Optimization Results")
            
            # Calculate metrics
            total_weight = st.session_state['solution']['total_weight']
            max_weight = st.session_state['truck']['max_weight']
            weight_percent = (total_weight / max_weight) * 100
            
            total_volume = st.session_state['solution']['total_volume']
            truck_volume = st.session_state['truck']['length'] * st.session_state['truck']['width'] * st.session_state['truck']['height']
            volume_percent = (total_volume / truck_volume) * 100
            
            cols = st.columns(3)
            cols[0].metric("Total Weight", 
                          f"{total_weight:.1f} kg / {max_weight:.1f} kg",
                          f"{weight_percent:.1f}%")
            cols[1].metric("Total Volume", 
                          f"{total_volume:.1f} m³ / {truck_volume:.1f} m³",
                          f"{volume_percent:.1f}%")
            cols[2].metric("Status", st.session_state['solution']['status'])
            
            tab1, tab2 = st.tabs(["3D Visualization", "Item Positions"])
            
            with tab1:
                fig = create_3d_box_visualization(st.session_state['truck'], st.session_state['solution']['items'])
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                solution_df = pd.DataFrame(st.session_state['solution']['items'])
                st.dataframe(solution_df.sort_values('priority'))
                
                # Download button
                csv = solution_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="truck_loading_solution.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.header("Visualization")
            st.info("Configure your items and truck specifications, then click 'Optimize Loading' to see results.")
            
            # Show empty truck visualization
            fig = create_3d_box_visualization(st.session_state['truck'], [])

if __name__ == "__main__":
    main()  