# local packages:
from model_funcs_mutag import build_model_mutag
from data_funcs import get_dataset


def main_train_test_validate_mutag():
    datasetname = 'MUTAG'
    dataset, DSpars, train_loader, val_loader, test_loader = get_dataset(
        datasetname=datasetname, report=True)

    gn_model, optimizer, loss_function = build_model_mutag(
        dataset, report=False)

    #train, validate, test
    train_val_test_mutag(gn_model, train_loader,
                            val_loader, test_loader, loss_function, optimizer)

    print('All tasks complete')

def train_val_test_mutag(gn_model, train_loader, val_loader, test_loader, loss_function, optimizer):
    for epoch in range(1, 70):
        print(f'Epoch: {epoch:03d}', end=':')
        train_mutag(gn_model, train_loader, loss_function, optimizer)
        train_acc = test_mutag(gn_model, train_loader)
        val_acc = test_mutag(gn_model, val_loader)
        print(
            f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}'
        )

    # do testing
    test_acc = test_mutag(gn_model, test_loader)
    print(f'Test Acc: {test_acc:.4f}')


def train_mutag(gn_model, train_loader, loss_function, optimizer):
    gn_model.train()
    batch=0
    for data in train_loader:  # Iterate over the batches of grahs
        batch +=1
        #print(f'\n -B{batch:01d}_data: {data}')
        #print(f'\n -B{batch:01d}_edgei[0][:10]: {data.edge_index[0][:10]}')
        #print(f'\n -B{batch:01d}_edgei[1][:10]: {data.edge_index[1][:10]}')
        out = gn_model(data.x, data.edge_attr, data.u, data.edge_index,
                       data.batch)  # Forward pass(es)
        loss = loss_function(out, data.y)  # Compute the loss
        loss.backward()  # Compute the gradients
        optimizer.step()  # Update the weights based on the computed gradients
        optimizer.zero_grad()  # Clear the computed gradients


def test_mutag(gn_model, loader):
    gn_model.eval()

    correct = 0
    for data in loader:
        # Iterate over the batches
        out = gn_model(data.x, data.edge_attr, data.u, data.edge_index,
                       data.batch)
        # Predict the labels using the label with the highest probability
        pred = out.argmax(dim=1)
        # Check against the ground truth
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)  # Compute accuracy


if __name__ == '__main__':
    main_train_test_validate_mutag()
