# Frontend - ChurnPredict

A Next.js-based web application for customer churn prediction using FastAPI backend.

## Features

### Single Customer Prediction
- Enter customer details through an intuitive form
- Get instant churn prediction with probability and confidence score
- View identified risk factors
- Receive data-driven retention recommendations

### Batch Prediction
- Upload multiple customer records via CSV
- Process predictions in bulk
- Download results as CSV for further analysis
- View summary statistics

### Model Insights
- View model performance metrics (Accuracy, Precision, Recall, F1 Score)
- Analyze feature importance
- Understand which factors contribute to churn predictions

### Interactive Dashboard
- Real-time API connection status indicator
- Responsive UI with dark/light theme support
- Comprehensive error handling and user feedback
- Tabbed interface for organized navigation

## Setup

### Prerequisites
- Node.js 18+ and npm/pnpm
- FastAPI backend running on `http://localhost:8000`

### Installation

```bash
cd frontend
npm install
# or
pnpm install
```

### Configuration

Create/update `.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Development

```bash
npm run dev
```

The application will be available at `http://localhost:3000`

### Build & Production

```bash
npm run build
npm start
```

## API Integration

The frontend communicates with the FastAPI backend via the `api.ts` service layer.

### Key Endpoints Used

- `GET /health` - Check API health
- `GET /model/info` - Get model metrics
- `GET /model/feature-importance` - Get feature importance
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch prediction

### API Service (`lib/api.ts`)

The `ChurnPredictionAPI` class provides methods for all API interactions:

```typescript
import { api } from '@/lib/api';

// Check health
const health = await api.checkHealth();

// Get model info
const info = await api.getModelInfo();

// Make prediction
const result = await api.predict(customerData);

// Batch prediction
const batchResult = await api.predictBatch(customers);
```

## Project Structure

```
frontend/
├── app/
│   ├── page.tsx          # Main dashboard
│   ├── layout.tsx        # Root layout
│   └── globals.css       # Global styles
├── components/
│   ├── customer-form.tsx       # Customer input form
│   ├── batch-prediction.tsx    # Batch prediction UI
│   ├── prediction-result.tsx   # Single prediction display
│   ├── feature-importance.tsx  # Feature importance chart
│   ├── model-metrics.tsx       # Model performance metrics
│   └── ui/                     # Shadcn/ui components
├── lib/
│   ├── api.ts           # API service layer
│   └── utils.ts         # Utility functions
└── styles/
    └── globals.css      # Global CSS
```

## Component Documentation

### CustomerForm
Comprehensive form for entering customer telco service details:
- Demographics (gender, senior citizen status, partner/dependents)
- Account information (tenure, charges, contract, payment method)
- Phone & Internet services
- Internet add-ons (security, backup, device protection, tech support, streaming)

### BatchPrediction
CSV-based bulk prediction interface:
- Paste CSV data
- Process multiple customers
- Download results

### PredictionResult
Displays single prediction with:
- Churn prediction (Yes/No)
- Probability and confidence
- Risk factors

### FeatureImportance
Bar chart showing which features most influence churn prediction

### ModelMetrics
Performance metrics display:
- Accuracy
- Precision
- Recall
- F1 Score

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | FastAPI backend URL | http://localhost:8000 |

## Error Handling

The frontend includes comprehensive error handling:
- API connection failures display user-friendly messages
- Missing model alerts guide users
- Form validation prevents invalid submissions
- Batch processing errors are detailed per-row

## Troubleshooting

### "Cannot connect to API"
- Ensure FastAPI backend is running on `http://localhost:8000`
- Check `NEXT_PUBLIC_API_URL` in `.env.local`
- Verify CORS is enabled in FastAPI backend

### "Model not loaded"
- Run the training script first: `python train.py`
- Check that model files exist in `models/` directory

### Form fields not updating
- Clear browser cache
- Restart development server
- Check browser console for errors

## Frontend Configuration Notes

- TypeScript errors are ignored during build (for quick iteration)
- Images are unoptimized (suitable for development)
- CORS is handled by the FastAPI backend
- The frontend uses Shadcn/ui components with Tailwind CSS

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## Development Tips

1. **API Testing**: Use the `/docs` endpoint on FastAPI backend for interactive API testing
2. **Form Validation**: All required fields must be filled before prediction
3. **Batch Processing**: CSV must have header row matching API schema
4. **State Management**: Uses React hooks (useState, useEffect, useCallback)
5. **Error States**: Always check `apiError` state before rendering predictions
